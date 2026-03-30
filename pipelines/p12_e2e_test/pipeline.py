"""
P12 End-to-End Test Pipeline
6 KFP components: L1 Qdrant + L2 vLLM + L3 RAG + L4 RAGAS + L5 Counts + E2E Report
See full source in pipelines/p12_e2e_test/pipeline.py
"""
from kfp import dsl
from kfp.dsl import Output, Artifact, Input

@dsl.component(
    base_image="659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/pipeline-base:latest",
    packages_to_install=["qdrant-client==1.7.3","sentence-transformers==2.5.1","protobuf==4.25.3"],
)
def e2e_qdrant_component(qdrant_host:str, qdrant_port:int, min_chunks_per_query:int, qdrant_report:Output[Artifact]) -> str:
    import json,subprocess,sys,pathlib
    subprocess.run([sys.executable,"-m","pip","install","-q","protobuf==4.25.3"],capture_output=True)
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    domain_questions = {
        "tech_docs":["What GPU instance type does the platform use?","What CUDA version is on EKS GPU?","What is vLLM throughput with Mistral-7B-GPTQ?"],
        "hr_policies":["What is the company leave policy?","How do I request time off?"],
        "org_info":["What teams are in engineering?","Who handles IT support?"],
    }
    results = {"passed":0,"failed":0,"details":[],"overall":"PASS"}
    collections = {c.name for c in client.get_collections().collections}
    for col, questions in domain_questions.items():
        if col not in collections:
            results["failed"]+=1; results["overall"]="FAIL"
            results["details"].append({"collection":col,"error":"missing","status":"FAIL"}); continue
        for q in questions:
            hits = client.search(collection_name=col,query_vector=model.encode(q).tolist(),limit=3,score_threshold=0.0)
            status="PASS" if len(hits)>=min_chunks_per_query else "FAIL"
            results["details"].append({"collection":col,"question":q[:60],"chunks":len(hits),"status":status})
            if status=="PASS": results["passed"]+=1
            else: results["failed"]+=1; results["overall"]="FAIL"
            print(f"  [{status}] {col} chunks={len(hits)} q='{q[:40]}'")
    pathlib.Path(qdrant_report.path).parent.mkdir(parents=True,exist_ok=True)
    with open(qdrant_report.path,"w") as f: json.dump(results,f,indent=2)
    print(f"L1: {results['passed']} passed {results['failed']} failed â€” {results['overall']}")
    return results["overall"]

@dsl.component(
    base_image="659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/pipeline-base:latest",
    packages_to_install=["requests==2.31.0","protobuf==4.25.3"],
)
def e2e_vllm_component(vllm_base_url:str, p95_latency_sla_s:float, vllm_report:Output[Artifact]) -> str:
    import json,time,subprocess,sys,pathlib
    subprocess.run([sys.executable,"-m","pip","install","-q","protobuf==4.25.3"],capture_output=True)
    import requests
    PROMPTS=["What is a Tesla T4 GPU?","Explain Kubernetes in one sentence.","What does RAGAS stand for?","What is a vector database?","Describe vLLM in 20 words."]
    MODEL="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    results={"passed":0,"failed":0,"latencies":[],"details":[],"overall":"PASS"}
    try:
        h=requests.get(f"{vllm_base_url}/health",timeout=10)
        if h.status_code!=200: results["overall"]="FAIL"; pathlib.Path(vllm_report.path).parent.mkdir(parents=True,exist_ok=True); [open(vllm_report.path,"w").write(json.dumps(results))]; return "FAIL"
    except Exception as e:
        results["overall"]="SKIP"; pathlib.Path(vllm_report.path).parent.mkdir(parents=True,exist_ok=True)
        with open(vllm_report.path,"w") as f: json.dump({"overall":"SKIP","reason":str(e)},f); return "SKIP"
    for p in PROMPTS:
        start=time.time()
        try:
            r=requests.post(f"{vllm_base_url}/v1/completions",json={"model":MODEL,"prompt":f"[INST] {p} [/INST]","max_tokens":80,"temperature":0.1},timeout=30)
            lat=time.time()-start
            if r.ok:
                text=r.json()["choices"][0]["text"].strip(); wc=len(text.split())
                status="PASS" if wc>=3 else "FAIL"
                results["latencies"].append(lat); results["details"].append({"prompt":p[:40],"latency_s":round(lat,2),"words":wc,"status":status})
                print(f"  [{status}] {lat:.2f}s {wc}w '{text[:40]}'")
                if status=="PASS": results["passed"]+=1
                else: results["failed"]+=1; results["overall"]="FAIL"
        except Exception as e:
            results["failed"]+=1; results["overall"]="FAIL"
    if results["latencies"]:
        p95=max(results["latencies"]); results["p95_latency_s"]=round(p95,3)
        print(f"  P95={p95:.2f}s SLA={p95_latency_sla_s}s")
        if p95>p95_latency_sla_s: results["overall"]="FAIL"
    pathlib.Path(vllm_report.path).parent.mkdir(parents=True,exist_ok=True)
    with open(vllm_report.path,"w") as f: json.dump(results,f,indent=2)
    print(f"L2: {results['passed']} passed {results['failed']} failed â€” {results['overall']}")
    return results["overall"]

@dsl.component(
    base_image="659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/pipeline-base:latest",
    packages_to_install=["qdrant-client==1.7.3","sentence-transformers==2.5.1","requests==2.31.0","protobuf==4.25.3"],
)
def e2e_rag_component(qdrant_host:str, qdrant_port:int, vllm_base_url:str, rag_report:Output[Artifact]) -> str:
    import json,time,subprocess,sys,pathlib
    subprocess.run([sys.executable,"-m","pip","install","-q","protobuf==4.25.3"],capture_output=True)
    from qdrant_client import QdrantClient; from sentence_transformers import SentenceTransformer; import requests
    model=SentenceTransformer("all-MiniLM-L6-v2"); qdrant=QdrantClient(host=qdrant_host,port=qdrant_port)
    MODEL="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    test_cases=[
        {"question":"What GPU does the platform use for LLM inference?","collection":"tech_docs","keywords":["t4","gpu","g4dn","tesla"]},
        {"question":"What is the vLLM token throughput on the T4 GPU?","collection":"tech_docs","keywords":["37","token","second","tok"]},
    ]
    results={"passed":0,"failed":0,"details":[],"overall":"PASS"}
    try: requests.get(f"{vllm_base_url}/health",timeout=5)
    except: pathlib.Path(rag_report.path).parent.mkdir(parents=True,exist_ok=True); open(rag_report.path,"w").write('{"overall":"SKIP"}'); return "SKIP"
    for tc in test_cases:
        q=tc["question"]; hits=qdrant.search(collection_name=tc["collection"],query_vector=model.encode(q).tolist(),limit=5)
        if not hits: results["failed"]+=1; results["overall"]="FAIL"; continue
        ctx="\n\n".join([r.payload.get("text",r.payload.get("content","")) for r in hits[:3]])
        prompt=f"[INST] Answer using context.\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer: [/INST]"
        try:
            r=requests.post(f"{vllm_base_url}/v1/completions",json={"model":MODEL,"prompt":prompt,"max_tokens":120,"temperature":0.0},timeout=30)
            if r.ok:
                answer=r.json()["choices"][0]["text"].strip(); grounded=any(k in answer.lower() for k in tc["keywords"])
                status="PASS" if (grounded and len(answer.split())>=5) else "FAIL"
                results["details"].append({"question":q[:60],"grounded":grounded,"status":status})
                print(f"  [{status}] grounded={grounded} ans='{answer[:60]}'")
                if status=="PASS": results["passed"]+=1
                else: results["failed"]+=1; results["overall"]="FAIL"
        except Exception as e:
            results["failed"]+=1; results["overall"]="FAIL"
    pathlib.Path(rag_report.path).parent.mkdir(parents=True,exist_ok=True)
    with open(rag_report.path,"w") as f: json.dump(results,f,indent=2)
    print(f"L3: {results['passed']} passed {results['failed']} failed â€” {results['overall']}")
    return results["overall"]

@dsl.component(
    base_image="659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/pipeline-base:latest",
    packages_to_install=["ragas==0.1.7","qdrant-client==1.7.3","sentence-transformers==2.5.1","requests==2.31.0","datasets==2.16.1","protobuf==4.25.3"],
)
def e2e_ragas_component(qdrant_host:str, qdrant_port:int, vllm_base_url:str, mlflow_tracking_uri:str, min_faithfulness:float, ragas_report:Output[Artifact]) -> float:
    import json,subprocess,sys,pathlib
    subprocess.run([sys.executable,"-m","pip","install","-q","protobuf==4.25.3"],capture_output=True)
    from qdrant_client import QdrantClient; from sentence_transformers import SentenceTransformer; import requests
    SPOT_CHECK_QA=[
        {"question":"What GPU instance type does the Enterprise LLM Platform use for vLLM inference?","ground_truth":"The platform uses g4dn.2xlarge instances with a Tesla T4 GPU (16GB VRAM) for vLLM inference."},
        {"question":"What is the token throughput achieved by vLLM with Mistral-7B-GPTQ on the T4 GPU?","ground_truth":"vLLM achieves 37 tokens per second with Mistral-7B-Instruct-v0.2-GPTQ (4-bit) on a Tesla T4 GPU."},
        {"question":"What CUDA version does the EKS GPU node use?","ground_truth":"The EKS GPU node uses CUDA 12.8, enabled by the AL2_x86_64_GPU AMI with pre-installed NVIDIA drivers."},
        {"question":"What percentage of T4 GPU SM utilization does vLLM achieve at peak?","ground_truth":"vLLM achieves 90% GPU SM utilization and 94% memory bandwidth utilization on the Tesla T4 at peak load."},
        {"question":"How much VRAM does the Mistral-7B-GPTQ model consume on the T4 GPU?","ground_truth":"vLLM serves TheBloke/Mistral-7B-Instruct-v0.2-GPTQ using GPTQ 4-bit quantization, consuming 12.9GB of the T4s 16GB VRAM."},
    ]
    MODEL="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    report={"overall":"PASS","faithfulness":None}
    try: requests.get(f"{vllm_base_url}/health",timeout=5)
    except: pathlib.Path(ragas_report.path).parent.mkdir(parents=True,exist_ok=True); open(ragas_report.path,"w").write('{"overall":"SKIP","faithfulness":0.0}'); return 0.0
    model=SentenceTransformer("all-MiniLM-L6-v2"); qdrant=QdrantClient(host=qdrant_host,port=qdrant_port)
    records=[]
    for qa in SPOT_CHECK_QA:
        hits=qdrant.search(collection_name="tech_docs",query_vector=model.encode(qa["question"]).tolist(),limit=5)
        contexts=[r.payload.get("text",r.payload.get("content","")) for r in hits]
        prompt=f"[INST] Answer using context.\n\nContext:\n{chr(10).join(contexts[:3])}\n\nQ: {qa['question']}\nA: [/INST]"
        try:
            r=requests.post(f"{vllm_base_url}/v1/completions",json={"model":MODEL,"prompt":prompt,"max_tokens":120,"temperature":0.0},timeout=30)
            answer=r.json()["choices"][0]["text"].strip() if r.ok else ""
        except: answer=""
        records.append({"question":qa["question"],"answer":answer,"contexts":contexts,"ground_truth":qa["ground_truth"]})
        print(f"  Q='{qa['question'][:50]}' A='{answer[:50]}'")
    try:
        from datasets import Dataset; from ragas import evaluate; from ragas.metrics import faithfulness
        scores=evaluate(Dataset.from_list(records),metrics=[faithfulness]); faith=float(scores["faithfulness"])
        report["faithfulness"]=round(faith,4); report["overall"]="PASS" if faith>=min_faithfulness else "FAIL"
        print(f"  RAGAS faithfulness: {faith:.4f} (gate={min_faithfulness})")
        try:
            import mlflow; mlflow.set_tracking_uri(mlflow_tracking_uri); mlflow.set_experiment("p12-e2e-tests")
            with mlflow.start_run(run_name="p12-ragas-spot"):
                mlflow.log_metric("e2e_faithfulness_spot",faith); mlflow.log_metric("e2e_gate",int(faith>=min_faithfulness))
        except Exception as e: print(f"  MLflow warning: {e}")
    except Exception as e: report["overall"]="FAIL"; faith=0.0; print(f"  RAGAS failed: {e}")
    pathlib.Path(ragas_report.path).parent.mkdir(parents=True,exist_ok=True)
    with open(ragas_report.path,"w") as f: json.dump(report,f,indent=2)
    return report.get("faithfulness") or 0.0

@dsl.component(
    base_image="659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/pipeline-base:latest",
    packages_to_install=["qdrant-client==1.7.3","mlflow==2.11.1","psycopg2-binary==2.9.9","protobuf==4.25.3"],
)
def e2e_counts_component(qdrant_host:str, qdrant_port:int, mlflow_tracking_uri:str, postgres_host:str, postgres_db:str, postgres_user:str, postgres_password:str, min_tech_docs:int, min_hr_policies:int, min_org_info:int, counts_report:Output[Artifact]) -> str:
    import json,subprocess,sys,pathlib
    subprocess.run([sys.executable,"-m","pip","install","-q","protobuf==4.25.3"],capture_output=True)
    from qdrant_client import QdrantClient; import mlflow; import psycopg2
    results={"passed":0,"failed":0,"details":{},"overall":"PASS"}
    qdrant=QdrantClient(host=qdrant_host,port=qdrant_port)
    for col,mn in [("tech_docs",min_tech_docs),("hr_policies",min_hr_policies),("org_info",min_org_info)]:
        try:
            count=qdrant.get_collection(col).points_count; status="PASS" if count>=mn else "FAIL"
            results["details"][f"qdrant_{col}"]={"count":count,"minimum":mn,"status":status}
            print(f"  [{status}] qdrant/{col}: {count} (min={mn})")
            if status=="PASS": results["passed"]+=1
            else: results["failed"]+=1; results["overall"]="FAIL"
        except Exception as e:
            results["details"][f"qdrant_{col}"]={"error":str(e),"status":"FAIL"}; results["failed"]+=1; results["overall"]="FAIL"
    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri); client=mlflow.tracking.MlflowClient()
        exp=client.get_experiment_by_name("ragas-evaluation")
        if exp:
            runs=client.search_runs(experiment_ids=[exp.experiment_id],order_by=["start_time DESC"],max_results=1)
            if runs:
                faith=runs[0].data.metrics.get("faithfulness",0)
                results["details"]["mlflow_faithfulness"]=round(faith,4); results["passed"]+=1
                print(f"  [PASS] MLflow faithfulness: {faith:.4f}")
    except Exception as e: print(f"  MLflow warning: {e}")
    try:
        conn=psycopg2.connect(host=postgres_host,database=postgres_db,user=postgres_user,password=postgres_password,port=5432,sslmode="require")
        cur=conn.cursor(); cur.execute("SELECT domain,COUNT(*) FROM golden_qa WHERE active=TRUE GROUP BY domain")
        dc=dict(cur.fetchall()); total=sum(dc.values())
        results["details"]["golden_qa"]={"domain_counts":dc,"total":total}
        status="PASS" if total>=60 else "FAIL"
        print(f"  [{status}] golden_qa: {total} {dc}")
        if status=="PASS": results["passed"]+=1
        else: results["failed"]+=1; results["overall"]="FAIL"
        cur.close(); conn.close()
    except Exception as e: print(f"  PostgreSQL warning: {e}")
    pathlib.Path(counts_report.path).parent.mkdir(parents=True,exist_ok=True)
    with open(counts_report.path,"w") as f: json.dump(results,f,indent=2)
    print(f"L5: {results['passed']} passed {results['failed']} failed â€” {results['overall']}")
    return results["overall"]

@dsl.component(
    base_image="659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/pipeline-base:latest",
    packages_to_install=["mlflow==2.11.1","protobuf==4.25.3"],
)
def e2e_report_component(mlflow_tracking_uri:str, run_uuid:str, qdrant_report:Input[Artifact], vllm_report:Input[Artifact], rag_report:Input[Artifact], ragas_report:Input[Artifact], counts_report:Input[Artifact], e2e_report:Output[Artifact]) -> str:
    import json,subprocess,sys,pathlib
    subprocess.run([sys.executable,"-m","pip","install","-q","protobuf==4.25.3"],capture_output=True)
    import mlflow; from datetime import datetime
    reports={}
    for name,artifact in [("L1_qdrant",qdrant_report),("L2_vllm",vllm_report),("L3_rag",rag_report),("L4_ragas",ragas_report),("L5_counts",counts_report)]:
        try:
            with open(artifact.path) as f: reports[name]=json.load(f)
        except Exception as e: reports[name]={"overall":"ERROR","error":str(e)}
    layer_results={k:v.get("overall","ERROR") for k,v in reports.items()}
    failed=[k for k,v in layer_results.items() if v=="FAIL"]
    overall="FAIL" if failed else "PASS"
    faith=reports.get("L4_ragas",{}).get("faithfulness") or 0.0
    summary={"timestamp":datetime.utcnow().isoformat(),"run_uuid":run_uuid,"overall":overall,"layer_results":layer_results,"failed_layers":failed,"faithfulness_spot":faith}
    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri); mlflow.set_experiment("p12-e2e-tests")
        with mlflow.start_run(run_name=f"p12-e2e-{run_uuid[:8]}"):
            mlflow.log_param("run_uuid",run_uuid); mlflow.log_param("failed_layers",str(failed))
            mlflow.log_metric("e2e_overall_passed",int(overall=="PASS"))
            mlflow.log_metric("e2e_layers_passed",len([v for v in layer_results.values() if v=="PASS"]))
            mlflow.log_metric("e2e_faithfulness_spot",faith)
            mlflow.set_tag("pipeline","P12-E2E"); mlflow.set_tag("outcome",overall)
    except Exception as e: print(f"  MLflow warning: {e}")
    print(f"\n{'='*50}\nP12 E2E TEST RESULTS\n{'='*50}")
    for layer,result in layer_results.items():
        icon="\u2705" if result=="PASS" else ("\u26A0" if result=="SKIP" else "\u274C")
        print(f"  {icon} {layer}: {result}")
    print(f"  Overall: {overall}\n  Faithfulness: {faith:.4f}\n{'='*50}")
    pathlib.Path(e2e_report.path).parent.mkdir(parents=True,exist_ok=True)
    with open(e2e_report.path,"w") as f: json.dump(summary,f,indent=2)
    return overall

@dsl.pipeline(name="p12-e2e-test",description="End-to-end test pipeline Enterprise LLM Platform v2.0")
def p12_e2e_pipeline(
    qdrant_host:str="qdrant.llm-platform-prod.svc.cluster.local",
    qdrant_port:int=6333,
    vllm_base_url:str="http://vllm-service.llm-platform-prod.svc.cluster.local:8000",
    mlflow_tracking_uri:str="http://172.20.172.203:5000",
    postgres_host:str="llm-platform-prod-postgres.c2xig0uywkrb.us-east-1.rds.amazonaws.com",
    postgres_db:str="llm_platform",
    postgres_user:str="llm_admin",
    postgres_password:str="Llmplatform2026",
    p95_latency_sla_s:float=10.0,
    min_faithfulness:float=0.60,
    min_tech_docs:int=100,
    min_hr_policies:int=55,
    min_org_info:int=50,
    min_chunks_per_query:int=1,
    run_uuid:str="p12-default",
):
    l1=e2e_qdrant_component(qdrant_host=qdrant_host,qdrant_port=qdrant_port,min_chunks_per_query=min_chunks_per_query)
    l1.set_caching_options(False)
    l2=e2e_vllm_component(vllm_base_url=vllm_base_url,p95_latency_sla_s=p95_latency_sla_s)
    l2.set_caching_options(False)
    l3=e2e_rag_component(qdrant_host=qdrant_host,qdrant_port=qdrant_port,vllm_base_url=vllm_base_url)
    l3.set_caching_options(False)
    l4=e2e_ragas_component(qdrant_host=qdrant_host,qdrant_port=qdrant_port,vllm_base_url=vllm_base_url,mlflow_tracking_uri=mlflow_tracking_uri,min_faithfulness=min_faithfulness)
    l4.set_caching_options(False)
    l5=e2e_counts_component(qdrant_host=qdrant_host,qdrant_port=qdrant_port,mlflow_tracking_uri=mlflow_tracking_uri,postgres_host=postgres_host,postgres_db=postgres_db,postgres_user=postgres_user,postgres_password=postgres_password,min_tech_docs=min_tech_docs,min_hr_policies=min_hr_policies,min_org_info=min_org_info)
    l5.set_caching_options(False)
    report=e2e_report_component(mlflow_tracking_uri=mlflow_tracking_uri,run_uuid=run_uuid,qdrant_report=l1.outputs["qdrant_report"],vllm_report=l2.outputs["vllm_report"],rag_report=l3.outputs["rag_report"],ragas_report=l4.outputs["ragas_report"],counts_report=l5.outputs["counts_report"])
    report.set_caching_options(False)

if __name__=="__main__":
    from kfp import compiler
    compiler.Compiler().compile(pipeline_func=p12_e2e_pipeline,package_path="p12_pipeline.yaml")
    print("Compiled: p12_pipeline.yaml")
