f = r"C:\Users\jayes\OneDrive\Desktop\jathakam\ai\k8s-deployent\RAGAS-KUBEFLOW-MLOPS\pipelines\p09_ragas_evaluation\pipeline.py"
with open(f) as fh:
    c = fh.read()

# Find all task assignments and add set_caching_options after
tasks = [
    "coverage_task",
    "retrieval_task", 
    "answers_task",
    "ragas_task",
]

for task in tasks:
    old = f"{task}.set_display_name"
    # task may not have set_display_name, find end of assignment instead
    pass

# Simpler - add set_caching_options(False) to each component call
# by replacing the closing paren of each task assignment
replacements = [
    ('coverage_task = check_qa_coverage_component(', 'coverage_task = check_qa_coverage_component('),
]

# Just add enable_caching=False as pipeline parameter
old = "@pipeline("
new = "@pipeline("

# Actually set it in the YAML via compile options
print("Will use compile with disable caching")
print(c[c.find("def ragas_evaluation_pipeline"):c.find("def ragas_evaluation_pipeline")+200])
