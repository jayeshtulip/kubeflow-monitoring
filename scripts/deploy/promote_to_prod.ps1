# Manual production promotion script
param([string]$ImageTag, [string]$ApproverName)
if (-not $ImageTag) { throw 'ImageTag required' }
Write-Host "Promoting $ImageTag to production (approved by $ApproverName)..." -ForegroundColor Yellow
# Update image tag in GitOps repo and trigger ArgoCD sync
git -C $env:GITOPS_REPO_PATH tag $ImageTag
git -C $env:GITOPS_REPO_PATH push --tags
Write-Host 'ArgoCD will sync automatically. Monitor: kubectl get rollout -n llm-platform-prod' -ForegroundColor Green
