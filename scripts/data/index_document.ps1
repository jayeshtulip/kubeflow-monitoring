# Trigger P5 Data Indexing pipeline for a specific document
param([string]$DocumentPath, [string]$Collection = 'tech_docs')
Write-Host "Indexing $DocumentPath into $Collection..." -ForegroundColor Cyan
python -m src.storage.qdrant.indexer --path $DocumentPath --collection $Collection
