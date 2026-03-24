# Configure kubectl context for EKS
param([string]$Env = 'staging')
aws eks update-kubeconfig --name llm-platform-$Env --region us-east-1
kubectl config use-context llm-platform-$Env
Write-Host "kubectl configured for $Env" -ForegroundColor Green
