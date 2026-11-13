
$ErrorActionPreference = "Stop"

$confidences = @(0.10, 0.25, 0.40)
$nmsValues = @(0.70)

foreach ($nms in $nmsValues) {
    foreach ($confidence in $confidences) {
        Write-Host "`n========================================"
        Write-Host "Running confidence=$confidence, NMS IoU=$nms"
        Write-Host "========================================`n"

        python -m scripts.evaluate_detector `
            --confidence $confidence `
            --nms-iou $nms

        if ($LASTEXITCODE -ne 0) {
            Write-Error "Experiment failed: confidence=$confidence, NMS=$nms"
            exit $LASTEXITCODE
        }
    }
}

Write-Host "`nAll remaining experiments completed successfully."
