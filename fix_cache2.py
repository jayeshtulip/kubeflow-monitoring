f = r"C:\Users\jayes\OneDrive\Desktop\jathakam\ai\k8s-deployent\RAGAS-KUBEFLOW-MLOPS\pipelines\p09_ragas_evaluation\pipeline.py"
with open(f) as fh:
    c = fh.read()

# Add .set_caching_options(False) after each task closing
replacements = [
    ("coverage_task = check_qa_coverage_component(", True),
    ("retrieval_task = retrieve_contexts_component(", True),
    ("answers_task = generate_answers_component(", True),
    ("ragas_task = ragas_score_component(", True),
]

# Find each task and add caching disable after its block
for task_name in ["coverage_task", "retrieval_task", "answers_task", "ragas_task"]:
    old = f"    {task_name}.set_display_name"
    if old in c:
        c = c.replace(old, f"    {task_name}.set_caching_options(False)\n    {task_name}.set_display_name")
    else:
        # find the task assignment end and add after
        import re
        pattern = f"({task_name} = [^(]+\\([^)]+\\))"
        # Just insert after the assignment line
        lines = c.split("\n")
        new_lines = []
        for line in lines:
            new_lines.append(line)
            if f"    {task_name} = " in line and line.strip().endswith(")"):
                new_lines.append(f"    {task_name}.set_caching_options(False)")
        c = "\n".join(new_lines)

with open(f, 'w') as fh:
    fh.write(c)
print("Done - check result")
