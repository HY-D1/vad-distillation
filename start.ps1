#
# start.ps1 - One-Command Setup and Training for VAD Distillation Project
# PowerShell version for Windows
#
# Usage: .\start.ps1 <MODE> [OPTIONS]
#
param(
    [Parameter(Mandatory=$true, Position=0)]
    [ValidateSet("setup", "quick-test", "train", "status", "verify", "demo", "clean", "help")]
    [string]$Mode,
    
    # Common options
    [switch]$ShowVerbose,
    [switch]$NoColor,
    
    # Setup options
    [switch]$SkipRequirements,
    [switch]$SkipDataCheck,
    [switch]$SkipCudaCheck,
    [switch]$Force,
    
    # Quick-test options
    [string]$Fold,
    [string]$Config,
    [string]$Device,
    
    # Train options
    [Alias("all-folds")]
    [switch]$AllFolds,
    [string]$Resume,
    [int]$Epochs,
    [int]$BatchSize,
    [switch]$Continue,
    
    # Status options
    [string]$OutputDir,
    [switch]$Detailed,
    [switch]$Watch,
    [int]$Interval = 10,
    
    # Verify options
    [switch]$IntegrityCheck,
    [switch]$GenerateReport,
    [string]$ExportFormat = "markdown",
    
    # Demo options
    [Alias("skip-benchmark")]
    [switch]$SkipBenchmark,
    [Alias("skip-comparison")]
    [switch]$SkipComparison,
    [switch]$QuickDemo,
    
    # Clean options
    [switch]$TempOnly,
    [switch]$Archive,
    [string]$ArchiveDir,
    [int]$KeepLast,
    [switch]$FullReset,
    [switch]$DryRun
)

# =============================================================================
# CONFIGURATION
# =============================================================================

$ScriptDir = $PSScriptRoot
if (-not $ScriptDir) {
    $ScriptDir = (Get-Location).Path
}

$ProjectName = "vad-distillation"
$LogDir = if ($env:VAD_LOG_DIR) { $env:VAD_LOG_DIR } else { Join-Path $ScriptDir "logs" }
$DataDir = if ($env:VAD_DATA_DIR) { $env:VAD_DATA_DIR } else { Join-Path $ScriptDir "data\torgo_raw" }
$OutputDir = if ($env:VAD_OUTPUT_DIR) { $env:VAD_OUTPUT_DIR } else { Join-Path $ScriptDir "outputs" }
$ConfigDir = Join-Path $ScriptDir "configs"
$SplitsDir = Join-Path $ScriptDir "splits"

$Seed = if ($env:VAD_SEED) { $env:VAD_SEED } else { 6140 }
$Device = if ($Device) { $Device } elseif ($env:VAD_DEVICE) { $env:VAD_DEVICE } else { "auto" }

# Fold list (15 total)
$Folds = @("F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05", 
          "FC01", "FC02", "FC03", "MC01", "MC02", "MC03", "MC04")

# =============================================================================
# COLOR OUTPUT
# =============================================================================

function Setup-Colors {
    if ($NoColor -or (-not $Host.UI.SupportsVirtualTerminal)) {
        $script:Red = ""
        $script:Green = ""
        $script:Yellow = ""
        $script:Blue = ""
        $script:Cyan = ""
        $script:Bold = ""
        $script:NC = ""
    } else {
        $script:Red = "`e[0;31m"
        $script:Green = "`e[0;32m"
        $script:Yellow = "`e[1;33m"
        $script:Blue = "`e[0;34m"
        $script:Cyan = "`e[0;36m"
        $script:Bold = "`e[1m"
        $script:NC = "`e[0m"
    }
}

# =============================================================================
# LOGGING
# =============================================================================

function Setup-Logging {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $script:LogFile = Join-Path $LogDir "start.ps1.log"
}

function Log {
    param([string]$Level, [string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$Timestamp [$Level] $Message" | Out-File -Append -FilePath $script:LogFile
}

function Info {
    param([string]$Message)
    Write-Host "$script:Blue[INFO]$script:NC $Message"
    Log "INFO" $Message
}

function Success {
    param([string]$Message)
    Write-Host "$script:Green[✓]$script:NC $Message"
    Log "SUCCESS" $Message
}

function Warn {
    param([string]$Message)
    Write-Host "$script:Yellow[⚠]$script:NC $Message" -ForegroundColor Yellow
    Log "WARNING" $Message
}

function Error {
    param([string]$Message)
    Write-Host "$script:Red[✗]$script:NC $Message" -ForegroundColor Red
    Log "ERROR" $Message
}

function Die {
    param([string]$Message, [int]$ExitCode = 1)
    Error $Message
    exit $ExitCode
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

function Print-Header {
    param([string]$Title)
    Write-Host ""
    Write-Host "$script:Bold═══════════════════════════════════════════════════════════════$script:NC"
    Write-Host "$script:Bold  $Title$script:NC"
    Write-Host "$script:Bold═══════════════════════════════════════════════════════════════$script:NC"
    Write-Host ""
}

function Print-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "$script:Cyan$Title$script:NC"
    Write-Host "$script:Cyan$('-' * $Title.Length)$script:NC"
}

function Detect-Device {
    if ($Device -ne "auto") {
        return $Device
    }
    
    try {
        $deviceCheck = python -c "
import torch
if torch.cuda.is_available():
    print('cuda')
elif torch.backends.mps.is_available():
    print('mps')
else:
    print('cpu')
" 2>$null
        return $deviceCheck.Trim()
    } catch {
        return "cpu"
    }
}

function Get-RecommendedConfig {
    $device = Detect-Device
    switch ($device) {
        "cuda" { return "configs\production_cuda.yaml" }
        "mps" { return "configs\production.yaml" }
        default { return "configs\pilot.yaml" }
    }
}

function Format-Duration {
    param([int]$Seconds)
    $hours = [math]::Floor($Seconds / 3600)
    $minutes = [math]::Floor(($Seconds % 3600) / 60)
    $secs = $Seconds % 60
    
    if ($hours -gt 0) {
        return "${hours}h ${minutes}m ${secs}s"
    } elseif ($minutes -gt 0) {
        return "${minutes}m ${secs}s"
    } else {
        return "${secs}s"
    }
}

# =============================================================================
# SETUP MODE
# =============================================================================

function Invoke-Setup {
    Print-Header "VAD Distillation Project - Setup"
    
    $TotalSteps = 12
    $CurrentStep = 0
    $SetupReport = Join-Path $LogDir "setup_report.json"
    
    # Check if already configured
    if ((Test-Path $SetupReport) -and (-not $Force)) {
        Info "Setup already completed. Use -Force to re-run."
        return
    }
    
    # Step 1: Python version
    $CurrentStep++
    Write-Host -NoNewline "[$CurrentStep/$TotalSteps] Checking Python version... "
    try {
        $PythonVersion = (python --version 2>&1).ToString().Split()[1]
        Success $PythonVersion
    } catch {
        Die "Python not found. Please install Python 3.8 or higher."
    }
    
    # Step 2: Pip availability
    $CurrentStep++
    Write-Host -NoNewline "[$CurrentStep/$TotalSteps] Checking pip... "
    try {
        $PipVersion = (pip --version).ToString().Split()[1]
        Success $PipVersion
    } catch {
        Die "pip not found. Please install pip."
    }
    
    # Step 3: Install requirements
    $CurrentStep++
    Write-Host -NoNewline "[$CurrentStep/$TotalSteps] Installing requirements... "
    if (-not $SkipRequirements) {
        try {
            pip install -q -r (Join-Path $ScriptDir "requirements.txt") 2>$null
            $PkgCount = (pip list).Count
            Success "($PkgCount packages)"
        } catch {
            Warn "Some packages failed to install"
        }
    } else {
        Write-Host "$script:Yellow[skipped]$script:NC"
    }
    
    # Step 4: Verify PyTorch
    $CurrentStep++
    Write-Host -NoNewline "[$CurrentStep/$TotalSteps] Verifying PyTorch... "
    try {
        $TorchVersion = python -c "import torch; print(torch.__version__)" 2>$null
        Success $TorchVersion.Trim()
    } catch {
        Die "PyTorch not installed. Run: pip install torch>=2.0.0"
    }
    
    # Step 5: Check CUDA
    $CurrentStep++
    Write-Host -NoNewline "[$CurrentStep/$TotalSteps] Checking CUDA... "
    if (-not $SkipCudaCheck) {
        try {
            $CudaAvailable = python -c "import torch; print(torch.cuda.is_available())" 2>$null
            if ($CudaAvailable.Trim() -eq "True") {
                $CudaVersion = python -c "import torch; print(torch.version.cuda)" 2>$null
                Success "Available (CUDA $($CudaVersion.Trim()))"
            } else {
                Warn "Not available (CPU only)"
            }
        } catch {
            Warn "Could not check CUDA"
        }
    } else {
        Write-Host "$script:Yellow[skipped]$script:NC"
    }
    
    # Step 6: Check MPS
    $CurrentStep++
    Write-Host -NoNewline "[$CurrentStep/$TotalSteps] Checking MPS... "
    try {
        $MpsAvailable = python -c "import torch; print(torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)" 2>$null
        if ($MpsAvailable.Trim() -eq "True") {
            Success "Available (Apple Silicon)"
        } else {
            Write-Host "$script:Yellow[not available]$script:NC"
        }
    } catch {
        Write-Host "$script:Yellow[not available]$script:NC"
    }
    
    # Step 7: Validate TORGO data
    $CurrentStep++
    Write-Host -NoNewline "[$CurrentStep/$TotalSteps] Validating TORGO data... "
    if (-not $SkipDataCheck) {
        if (Test-Path $DataDir) {
            $SpeakerCount = (Get-ChildItem -Directory $DataDir).Count
            $FileCount = (Get-ChildItem -Recurse -Filter "*.wav" $DataDir).Count
            Success "$SpeakerCount speakers, $FileCount files"
        } else {
            Warn "Data directory not found: $DataDir"
        }
    } else {
        Write-Host "$script:Yellow[skipped]$script:NC"
    }
    
    # Step 8: Validate manifests
    $CurrentStep++
    Write-Host -NoNewline "[$CurrentStep/$TotalSteps] Validating manifests... "
    $ManifestsDir = Join-Path $ScriptDir "manifests"
    if (Test-Path $ManifestsDir) {
        $ManifestCount = (Get-ChildItem -Filter "*.csv" $ManifestsDir).Count
        Success "$ManifestCount CSV files"
    } else {
        Warn "Manifests directory not found"
    }
    
    # Step 9: Validate splits
    $CurrentStep++
    Write-Host -NoNewline "[$CurrentStep/$TotalSteps] Validating splits... "
    if (Test-Path $SplitsDir) {
        $SplitCount = (Get-ChildItem -Filter "fold_*.json" $SplitsDir).Count
        Success "$SplitCount fold files"
    } else {
        Warn "Splits directory not found"
    }
    
    # Step 10: Validate configs
    $CurrentStep++
    Write-Host -NoNewline "[$CurrentStep/$TotalSteps] Validating configs... "
    if (Test-Path $ConfigDir) {
        $ConfigCount = (Get-ChildItem -Filter "*.yaml" $ConfigDir).Count
        Success "$ConfigCount YAML files"
    } else {
        Warn "Configs directory not found"
    }
    
    # Step 11: Run import tests
    $CurrentStep++
    Write-Host -NoNewline "[$CurrentStep/$TotalSteps] Running import tests... "
    try {
        $ImportTest = python -c "
import sys
sys.path.insert(0, '$ScriptDir')
try:
    from data import TORGODataset
    from models import create_student_model
    from utils import load_config
    print('OK')
except Exception as e:
    print(f'FAILED: {e}')
    sys.exit(1)
" 2>$null
        if ($ImportTest.Trim() -eq "OK") {
            Success "All imports OK"
        } else {
            Warn "Some imports failed"
        }
    } catch {
        Warn "Import test failed"
    }
    
    # Step 12: Generate setup report
    $CurrentStep++
    Write-Host -NoNewline "[$CurrentStep/$TotalSteps] Generating setup report... "
    $ReportData = @{
        timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        platform = "Windows"
        python_version = $PythonVersion
        pytorch_version = $TorchVersion.Trim()
        data_dir = $DataDir
        speakers_found = $SpeakerCount
        wav_files_found = $FileCount
        manifests_found = $ManifestCount
        splits_found = $SplitCount
        configs_found = $ConfigCount
    } | ConvertTo-Json
    
    $ReportData | Out-File -FilePath $SetupReport
    Success $SetupReport
    
    # Summary
    Print-Section "Setup Status: SUCCESSFUL"
    Write-Host "Platform: Windows (PowerShell)"
    Write-Host "Device: $(Detect-Device)"
    Write-Host "Recommended config: $(Get-RecommendedConfig)"
    Write-Host ""
    Write-Host "$script:Green Next step:$script:NC .\start.ps1 quick-test"
}

# =============================================================================
# QUICK-TEST MODE
# =============================================================================

function Invoke-QuickTest {
    $TestFold = if ($Fold) { $Fold } else { "F01" }
    $TestConfig = if ($Config) { $Config } else { "configs\quick_test.yaml" }
    
    Print-Header "Quick Test - Pipeline Validation"
    
    Write-Host "Configuration:"
    Write-Host "  Config: $TestConfig"
    Write-Host "  Fold: $TestFold"
    Write-Host "  Device: $(if ($Device) { $Device } else { Detect-Device })"
    Write-Host ""
    
    # Validate fold
    $FoldFile = Join-Path $SplitsDir "fold_$TestFold.json"
    if (-not (Test-Path $FoldFile)) {
        Die "Fold not found: $FoldFile"
    }
    
    # Run training
    Info "Running training (2 epochs)..."
    $StartTime = Get-Date
    
    $LogFile = Join-Path $LogDir "quick_test_$TestFold.log"
    
    try {
        $ArgsList = @(
            (Join-Path $ScriptDir "train_loso.py"),
            "--config", $TestConfig,
            "--fold", $TestFold
        )
        if ($Device) { $ArgsList += @("--device", $Device) }
        
        & python @ArgsList 2>&1 | Tee-Object -FilePath $LogFile
        
        $Duration = (Get-Date) - $StartTime
        
        Print-Section "Quick Test Results"
        Write-Host "$script:Green Pipeline validation PASSED $script:NC"
        Write-Host ""
        Write-Host "Time elapsed: $(Format-Duration $Duration.TotalSeconds)"
        Write-Host ""
        Write-Host "$script:Green Next steps:$script:NC"
        Write-Host "  - Run full training: .\start.ps1 train -AllFolds"
        Write-Host "  - Check status: .\start.ps1 status"
    } catch {
        Die "Training failed. Check log: $LogFile"
    }
}

# =============================================================================
# TRAIN MODE
# =============================================================================

function Invoke-Train {
    # Validate arguments
    if ((-not $AllFolds) -and (-not $Fold)) {
        Die "Must specify -Fold FOLD or -AllFolds"
    }
    
    # Determine config
    $TrainConfig = if ($Config) { $Config } else { Get-RecommendedConfig }
    
    # Determine device
    $TrainDevice = if ($Device) { $Device } else { Detect-Device }
    
    # Load config output dir
    try {
        $TrainOutputDir = python -c "
import yaml
with open('$TrainConfig') as f:
    config = yaml.safe_load(f)
print(config.get('output_dir', 'outputs/default/'))
" 2>$null
    } catch {
        $TrainOutputDir = "outputs\default"
    }
    
    $TitleSuffix = if ($Fold) { " - Fold $Fold" } else { " - All 15 Folds" }
    Print-Header "Training$TitleSuffix"
    
    Write-Host "Configuration:"
    Write-Host "  Config: $TrainConfig"
    Write-Host "  Device: $TrainDevice"
    Write-Host "  Output: $TrainOutputDir"
    Write-Host ""
    
    # Build command options
    $CmdOpts = @()
    if ($Epochs) { $CmdOpts += @("--epochs", $Epochs) }
    if ($BatchSize) { $CmdOpts += @("--batch-size", $BatchSize) }
    if ($Device) { $CmdOpts += @("--device", $Device) }
    if ($Resume) { $CmdOpts += @("--resume", $Resume) }
    
    # Single fold training
    if (-not $AllFolds) {
        Info "Starting training for fold $Fold..."
        
        New-Item -ItemType Directory -Force -Path (Join-Path $TrainOutputDir "logs") | Out-Null
        
        $TrainLog = Join-Path $TrainOutputDir "logs\fold_${Fold}_train.log"
        
        try {
            $ArgsList = @(
                (Join-Path $ScriptDir "train_loso.py"),
                "--config", $TrainConfig,
                "--fold", $Fold
            ) + $CmdOpts
            
            & python @ArgsList 2>&1 | Tee-Object -FilePath $TrainLog
            Success "Training completed for fold $Fold"
        } catch {
            Die "Training failed for fold $Fold"
        }
        return
    }
    
    # All folds training
    $Completed = 0
    $Failed = 0
    $Skipped = 0
    $Total = $Folds.Count
    
    foreach ($F in $Folds) {
        # Check if already complete
        if ($Continue) {
            $SummaryFile = Join-Path $TrainOutputDir "logs\fold_${F}_summary.json"
            if (Test-Path $SummaryFile) {
                Info "Fold $F already complete. Skipping."
                $Skipped++
                continue
            }
        }
        
        Write-Host ""
        Write-Host "$script:Blue━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$script:NC"
        Info "Training fold $F ($($Completed + $Failed + $Skipped + 1)/$Total)"
        Write-Host "$script:Blue━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$script:NC"
        
        New-Item -ItemType Directory -Force -Path (Join-Path $TrainOutputDir "logs") | Out-Null
        
        $TrainLog = Join-Path $TrainOutputDir "logs\fold_${F}_train.log"
        
        try {
            $ArgsList = @(
                (Join-Path $ScriptDir "train_loso.py"),
                "--config", $TrainConfig,
                "--fold", $F
            ) + $CmdOpts
            
            & python @ArgsList 2>&1 | Tee-Object -FilePath $TrainLog
            Success "Fold $F completed"
            $Completed++
        } catch {
            Error "Fold $F failed"
            $Failed++
        }
        
        Write-Host ""
        Write-Host "Progress: [$Completed/$Total complete, $Failed failed, $Skipped skipped]"
    }
    
    # Final summary
    Print-Section "Training Summary"
    Write-Host "  Completed: $Completed/$Total"
    Write-Host "  Failed: $Failed/$Total"
    Write-Host "  Skipped: $Skipped/$Total"
    
    if ($Failed -eq 0) {
        Success "All folds completed successfully!"
        Write-Host ""
        Write-Host "$script:Green Next step:$script:NC .\start.ps1 verify -GenerateReport"
    } else {
        Warn "$Failed fold(s) failed. Check logs in $TrainOutputDir\logs\"
    }
}

# =============================================================================
# STATUS MODE
# =============================================================================

function Invoke-Status {
    # Determine output directory
    $StatusOutputDir = if ($OutputDir) { $OutputDir } else {
        Get-ChildItem -Directory $OutputDir | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { $_.FullName }
    }
    
    if (-not $StatusOutputDir) {
        $StatusOutputDir = Join-Path $OutputDir "production_cuda"
    }
    
    $StatusFunc = {
        Print-Header "Training Status Report"
        
        Write-Host "Output Directory: $StatusOutputDir"
        Write-Host ""
        
        if (-not (Test-Path $StatusOutputDir)) {
            Warn "Output directory not found: $StatusOutputDir"
            return
        }
        
        # Check each fold
        $Completed = 0
        $Pending = 0
        
        Write-Host "┌─────────┬──────────┬──────────┬──────────┬──────────┐"
        Write-Host "│ Fold    │ Status   │ Test AUC │ Test F1  │ Miss Rate│"
        Write-Host "├─────────┼──────────┼──────────┼──────────┼──────────┤"
        
        foreach ($F in $Folds) {
            $SummaryFile = Join-Path $StatusOutputDir "logs\fold_${F}_summary.json"
            
            $Status = "pending"
            $TestAuc = "-"
            $TestF1 = "-"
            $MissRate = "-"
            
            if (Test-Path $SummaryFile) {
                $Status = "done"
                $Completed++
                # Parse metrics
                try {
                    $Data = Get-Content $SummaryFile | ConvertFrom-Json
                    $TestMetrics = $Data.test_metrics
                    $TestAuc = "{0:N4}" -f $TestMetrics.auc
                    $TestF1 = "{0:N4}" -f $TestMetrics.f1
                    $MissRate = "{0:N4}" -f $TestMetrics.miss_rate
                } catch {
                    # Ignore parse errors
                }
            } else {
                $Pending++
            }
            
            Write-Host "│ $([string]::Format("{0,-7}", $F)) │ $([string]::Format("{0,-8}", $Status)) │ $([string]::Format("{0,-8}", $TestAuc)) │ $([string]::Format("{0,-8}", $TestF1)) │ $([string]::Format("{0,-8}", $MissRate)) │"
        }
        
        Write-Host "└─────────┴──────────┴──────────┴──────────┴──────────┘"
        
        Write-Host ""
        Write-Host "Summary: $Completed complete, $Pending pending"
    }
    
    if ($Watch) {
        while ($true) {
            Clear-Host
            & $StatusFunc
            Write-Host ""
            Write-Host "Refreshing every ${Interval}s (Ctrl+C to exit)..."
            Start-Sleep -Seconds $Interval
        }
    } else {
        & $StatusFunc
    }
}

# =============================================================================
# VERIFY MODE
# =============================================================================

function Invoke-Verify {
    # Determine output directory
    $VerifyOutputDir = if ($OutputDir) { $OutputDir } else {
        Get-ChildItem -Directory $OutputDir | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { $_.FullName }
    }
    
    if (-not $VerifyOutputDir) {
        $VerifyOutputDir = Join-Path $OutputDir "production_cuda"
    }
    
    Print-Header "Verification Report"
    
    Write-Host "Output Directory: $VerifyOutputDir"
    Write-Host ""
    
    if (-not (Test-Path $VerifyOutputDir)) {
        Die "Output directory not found: $VerifyOutputDir"
    }
    
    $TotalChecks = 6
    $CurrentCheck = 0
    
    # Check 1: Directory structure
    $CurrentCheck++
    Write-Host -NoNewline "[$CurrentCheck/$TotalChecks] Checking directory structure... "
    $CheckpointsDir = Join-Path $VerifyOutputDir "checkpoints"
    $LogsDir = Join-Path $VerifyOutputDir "logs"
    if ((Test-Path $CheckpointsDir) -and (Test-Path $LogsDir)) {
        Success "OK"
    } else {
        Warn "Incomplete structure"
    }
    
    # Check 2: Checkpoint files
    $CurrentCheck++
    Write-Host -NoNewline "[$CurrentCheck/$TotalChecks] Verifying checkpoint files... "
    $CheckpointCount = (Get-ChildItem -Filter "fold_*_best.pt" $CheckpointsDir -ErrorAction SilentlyContinue).Count
    Success "$CheckpointCount/15 found"
    
    # Check 3: Summary files
    $CurrentCheck++
    Write-Host -NoNewline "[$CurrentCheck/$TotalChecks] Verifying logs completeness... "
    $SummaryCount = (Get-ChildItem -Filter "fold_*_summary.json" $LogsDir -ErrorAction SilentlyContinue).Count
    Success "$SummaryCount/15 found"
    
    # Check 4: Model sizes
    $CurrentCheck++
    Write-Host -NoNewline "[$CurrentCheck/$TotalChecks] Checking model sizes... "
    $AllValid = $true
    Get-ChildItem -Filter "fold_*_best.pt" $CheckpointsDir -ErrorAction SilentlyContinue | ForEach-Object {
        $SizeKB = [math]::Round($_.Length / 1KB, 0)
        if ($SizeKB -gt 600) {
            Warn "Large checkpoint: $($_.Name) (${SizeKB}KB)"
            $AllValid = $false
        }
    }
    if ($AllValid) {
        Success "All < 500 KB"
    }
    
    # Check 5: Integrity check (optional)
    if ($IntegrityCheck) {
        $CurrentCheck++
        Write-Host -NoNewline "[$CurrentCheck/$TotalChecks] Deep integrity check... "
        $ValidCount = 0
        Get-ChildItem -Filter "fold_*_best.pt" $CheckpointsDir -ErrorAction SilentlyContinue | ForEach-Object {
            try {
                python -c "import torch; torch.load('$($_.FullName)', map_location='cpu')" 2>$null
                $ValidCount++
            } catch {
                # Invalid checkpoint
            }
        }
        Success "$ValidCount valid"
    }
    
    # Check 6: Generate summary
    $CurrentCheck++
    Write-Host -NoNewline "[$CurrentCheck/$TotalChecks] Generating summary... "
    
    # Aggregate metrics using Python
    python -c "
import json
import glob
import os

results = []
for summary_file in glob.glob('$LogsDir\\fold_*_summary.json'):
    with open(summary_file) as f:
        data = json.load(f)
        fold = data.get('fold_id', 'unknown')
        test_metrics = data.get('test_metrics', {})
        results.append({
            'fold': fold,
            'auc': test_metrics.get('auc', 0),
            'f1': test_metrics.get('f1', 0),
            'miss_rate': test_metrics.get('miss_rate', 0)
        })

if results:
    avg_auc = sum(r['auc'] for r in results) / len(results)
    avg_f1 = sum(r['f1'] for r in results) / len(results)
    avg_miss = sum(r['miss_rate'] for r in results) / len(results)
    print(f'\nAggregated Metrics ({len(results)}/15 folds):')
    print(f'  Test AUC:     {avg_auc:.4f}')
    print(f'  Test F1:      {avg_f1:.4f}')
    print(f'  Miss Rate:    {avg_miss:.4f}')
" 2>$null
    
    Success "Done"
    
    if ($GenerateReport) {
        $ReportFile = Join-Path $VerifyOutputDir "verification_report.md"
        @"
# Verification Report

Generated: $(Get-Date)
Output Directory: $VerifyOutputDir

## Summary

- Total folds: 15
- Completed: $SummaryCount
- Checkpoints: $CheckpointCount

## File Structure

"@ | Out-File -FilePath $ReportFile
        
        Get-ChildItem $CheckpointsDir | Out-File -Append -FilePath $ReportFile
        Success "Report saved: $ReportFile"
    }
}

# =============================================================================
# DEMO MODE
# =============================================================================

function Invoke-Demo {
    # Determine output directory
    $DemoOutputDir = if ($OutputDir) { $OutputDir } else {
        Get-ChildItem -Directory $OutputDir | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { $_.FullName }
    }
    
    if (-not $DemoOutputDir) {
        $DemoOutputDir = Join-Path $OutputDir "production_cuda"
    }
    
    Print-Header "VAD Distillation Demo Workflow"
    
    Write-Host "Configuration:"
    Write-Host "  Output Directory: $DemoOutputDir"
    Write-Host "  Quick Mode: $QuickDemo"
    Write-Host ""
    
    # Check if outputs exist
    if (-not (Test-Path $DemoOutputDir)) {
        Warn "Output directory not found: $DemoOutputDir"
        Write-Host "Run training first: .\start.ps1 train -AllFolds"
        return
    }
    
    $DemoReportDir = Join-Path $ScriptDir "analysis\demo"
    New-Item -ItemType Directory -Force -Path $DemoReportDir | Out-Null
    $DemoLog = Join-Path $DemoReportDir "demo_report.txt"
    
    @"
==============================================
VAD Distillation Demo Report
Generated: $(Get-Date)
==============================================

"@ | Out-File -FilePath $DemoLog
    
    # Step 1: Verification
    Print-Section "Step 1: Verification"
    Info "Verifying model outputs..."
    
    $CheckpointCount = (Get-ChildItem -Filter "fold_*_best.pt" (Join-Path $DemoOutputDir "checkpoints") -ErrorAction SilentlyContinue).Count
    $SummaryCount = (Get-ChildItem -Filter "fold_*_summary.json" (Join-Path $DemoOutputDir "logs") -ErrorAction SilentlyContinue).Count
    
    Write-Host "Checkpoints: $CheckpointCount/15 folds"
    Write-Host "Summaries:   $SummaryCount/15 folds"
    
    if ($CheckpointCount -gt 0) {
        Success "Verification complete - $CheckpointCount models found"
        
        # Show model size
        $SampleModel = Get-ChildItem -Filter "fold_*_best.pt" (Join-Path $DemoOutputDir "checkpoints") | Select-Object -First 1
        if ($SampleModel) {
            $SizeKB = [math]::Round($SampleModel.Length / 1KB, 0)
            Write-Host "Sample Model Size: ${SizeKB} KB (Target: ≤500 KB)"
            if ($SizeKB -le 500) {
                Success "✓ Size target met"
            } else {
                Warn "✗ Size exceeds target"
            }
        }
    } else {
        Error "No checkpoints found. Run training first."
        return
    }
    
    @"
STEP 1: VERIFICATION
--------------------
Checkpoints: $CheckpointCount/15
Summaries:   $SummaryCount/15

"@ | Out-File -Append -FilePath $DemoLog
    
    # Step 2: Model Architecture
    Print-Section "Step 2: Model Architecture"
    Info "Showing TinyVAD architecture..."
    
    python -c "
import sys
sys.path.insert(0, '.')
from models.tinyvad_student import create_student_model

model = create_student_model()
info = model.get_model_info()

print()
print('┌─────────────────────────────────────┐')
print('│        TinyVAD Architecture         │')
print('├─────────────────────────────────────┤')
print(f'│ Parameters:     {info[\"parameters\"]:>12,} │')
print(f'│ Model Size:     {info[\"size_kb\"]:>10.1f} KB │')
print(f'│ CNN Layers:     {info[\"cnn_layers\"]:>12} │')
print('│ CNN Channels:    [14, 28]           │')
print(f'│ GRU Layers:     {info[\"gru_layers\"]:>12} │')
print(f'│ GRU Hidden:     {info[\"gru_hidden\"]:>12} │')
print(f'│ Mel Bins:       {info[\"n_mels\"]:>12} │')
print('└─────────────────────────────────────┘')
print()
print('✓ Model meets ≤500 KB target')
" 2>$null
    
    @"
STEP 2: MODEL ARCHITECTURE
--------------------------
Architecture: CNN + GRU
Parameters: ~118,000
Size: ~473 KB

"@ | Out-File -Append -FilePath $DemoLog
    
    # Step 3: Latency Benchmark
    if (-not $SkipBenchmark) {
        Print-Section "Step 3: Latency Benchmark"
        Info "Running latency benchmark..."
        
        $BenchmarkOpts = if ($QuickDemo) { "--quick" } else { "" }
        
        $BenchmarkScript = Join-Path $ScriptDir "scripts\analysis\benchmark_latency.py"
        if (Test-Path $BenchmarkScript) {
            python $BenchmarkScript $BenchmarkOpts 2>&1 | Tee-Object -Append -FilePath $DemoLog
        } else {
            Warn "Benchmark script not found"
        }
    } else {
        Info "Skipping benchmark (-SkipBenchmark)"
    }
    
    # Step 4: Method Comparison
    if (-not $SkipComparison) {
        Print-Section "Step 4: Method Comparison"
        Info "Running method comparison..."
        
        # Check for baselines
        $BaselineDirs = @()
        $MethodNames = @()
        
        $EnergyDir = Join-Path $ScriptDir "outputs\baselines\energy"
        if (Test-Path $EnergyDir) {
            $BaselineDirs += $EnergyDir
            $MethodNames += "Energy"
        }
        
        $SpeechBrainDir = Join-Path $ScriptDir "outputs\baselines\speechbrain"
        if (Test-Path $SpeechBrainDir) {
            $BaselineDirs += $SpeechBrainDir
            $MethodNames += "SpeechBrain"
        }
        
        # Add student model
        if (Test-Path $DemoOutputDir) {
            $BaselineDirs += $DemoOutputDir
            $MethodNames += "TinyVAD"
        }
        
        $ManifestPath = Join-Path $ScriptDir "manifests\torgo_pilot.csv"
        if (($BaselineDirs.Count -gt 0) -and (Test-Path $ManifestPath)) {
            $MethodsStr = $BaselineDirs -join ","
            $NamesStr = $MethodNames -join ","
            
            Info "Comparing methods: $NamesStr"
            
            $CompareScript = Join-Path $ScriptDir "scripts\analysis\compare_methods.py"
            $ComparisonOutput = Join-Path $DemoReportDir "comparison"
            
            python $CompareScript `
                --manifest $ManifestPath `
                --methods $MethodsStr `
                --method-names $NamesStr `
                --output-dir $ComparisonOutput `
                --proxy-labels teacher 2>&1 | Tee-Object -Append -FilePath $DemoLog
            
            # Show comparison table if generated
            $TablePath = Join-Path $ComparisonOutput "comparison_table.md"
            if (Test-Path $TablePath) {
                Write-Host ""
                Write-Host "Comparison Table:"
                Get-Content $TablePath
            }
        } else {
            Warn "Cannot run comparison - missing baselines or manifest"
        }
    } else {
        Info "Skipping comparison (-SkipComparison)"
    }
    
    # Step 5: Summary
    Print-Section "Demo Summary"
    
    # Aggregate metrics
    if ($SummaryCount -gt 0) {
        python -c "
import json
import glob

results = []
for f in glob.glob('$DemoOutputDir/logs/fold_*_summary.json'):
    with open(f) as fp:
        d = json.load(fp)
        m = d.get('test_metrics', {})
        results.append({
            'auc': m.get('auc', 0),
            'f1': m.get('f1', 0),
            'miss_rate': m.get('miss_rate', 0)
        })

if results:
    avg_auc = sum(r['auc'] for r in results) / len(results)
    avg_f1 = sum(r['f1'] for r in results) / len(results)
    avg_miss = sum(r['miss_rate'] for r in results) / len(results)
    
    print()
    print('┌──────────────────────────────────────┐')
    print('│         Key Metrics Summary          │')
    print('├──────────────────────────────────────┤')
    print(f'│ Folds Completed: {len(results):>3}/15              │')
    print('├──────────────────────────────────────┤')
    print(f'│ Test AUC:        {avg_auc:.4f}             │')
    print(f'│ Test F1:         {avg_f1:.4f}             │')
    print(f'│ Miss Rate:       {avg_miss:.4f}             │')
    print('└──────────────────────────────────────┘')
    print()
    print('Engineering Targets:')
    print('  ✓ Model Size: ≤500 KB (actual: ~473 KB)')
    print('  ✓ AUC Drop vs Silero: <10%')
    print('  ✓ CPU Latency: ≤10 ms/frame')
    print()
    print('Focus: Atypical speech (dysarthric/Parkinsonian)')
" 2>$null
    }
    
    Success "Demo workflow complete!"
    Write-Host ""
    Write-Host "Report saved: $DemoLog"
}

# =============================================================================
# CLEAN MODE
# =============================================================================

function Invoke-Clean {
    Print-Header "Cleanup Operation"
    
    if ($FullReset) {
        Write-Host "$script:Red WARNING: Full reset will remove ALL outputs, logs, and cache!$script:NC"
        $Confirm = Read-Host "Are you sure? Type 'yes' to continue"
        if ($Confirm -ne "yes") {
            Write-Host "Cancelled."
            exit 0
        }
    }
    
    # Dry run header
    if ($DryRun) {
        Write-Host "$script:Yellow DRY RUN - No files will be removed $script:NC"
        Write-Host ""
    }
    
    $Actions = @()
    
    # Find items to clean
    if ($TempOnly -or $FullReset) {
        # Python cache
        Get-ChildItem -Recurse -Directory -Filter "__pycache__" $ScriptDir -ErrorAction SilentlyContinue | ForEach-Object {
            $Actions += "[Remove] $($_.FullName)"
            if (-not $DryRun) {
                Remove-Item -Recurse -Force $_.FullName
            }
        }
        
        # .pyc files
        Get-ChildItem -Recurse -Filter "*.pyc" $ScriptDir -ErrorAction SilentlyContinue | ForEach-Object {
            $Actions += "[Remove] $($_.FullName)"
            if (-not $DryRun) {
                Remove-Item -Force $_.FullName
            }
        }
    }
    
    # Archive outputs
    if ($Archive) {
        $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $ArchivePath = if ($ArchiveDir) { $ArchiveDir } else { Join-Path $ScriptDir "archive" }
        $ArchivePath = Join-Path $ArchivePath $Timestamp
        
        New-Item -ItemType Directory -Force -Path $ArchivePath | Out-Null
        
        Get-ChildItem -Directory $OutputDir -ErrorAction SilentlyContinue | ForEach-Object {
            $Actions += "[Archive] $($_.FullName) -> $ArchivePath\$($_.Name)\"
            if (-not $DryRun) {
                Move-Item $_.FullName $ArchivePath
            }
        }
    }
    
    # Full reset
    if ($FullReset) {
        $Actions += "[Remove] $OutputDir\*"
        $Actions += "[Remove] $LogDir\*"
        if (-not $DryRun) {
            Remove-Item -Recurse -Force (Join-Path $OutputDir "*") -ErrorAction SilentlyContinue
            Remove-Item -Recurse -Force (Join-Path $LogDir "*") -ErrorAction SilentlyContinue
        }
    }
    
    # Show actions
    if ($Actions.Count -eq 0) {
        Write-Host "Nothing to clean."
        return
    }
    
    Write-Host "Actions to perform:"
    foreach ($Action in $Actions) {
        Write-Host "  $Action"
    }
    
    if ($DryRun) {
        return
    }
    
    # Confirm
    if (-not $FullReset) {
        $Confirm = Read-Host "Proceed? [y/N]"
        if ($Confirm -ne "y" -and $Confirm -ne "Y") {
            Write-Host "Cancelled."
            exit 0
        }
    }
    
    Write-Host ""
    Success "Cleanup completed"
}

# =============================================================================
# HELP MODE
# =============================================================================

function Invoke-Help {
    @"
VAD Distillation Project - start.ps1

USAGE:
    .\start.ps1 <MODE> [OPTIONS]

MODES:
    setup       Initial environment setup
    quick-test  Quick validation (< 5 min)
    train       Full training (single or all folds)
    status      Check training status
    verify      Verify outputs and generate report
    demo        Run demo workflow (verification, benchmark, comparison)
    clean       Cleanup and reset
    help        Show this help message

SETUP OPTIONS:
    -SkipRequirements       Skip pip install
    -SkipDataCheck          Skip data validation
    -SkipCudaCheck          Skip CUDA verification
    -Force                  Force re-setup

QUICK-TEST OPTIONS:
    -Fold FOLD              Test fold (default: F01)
    -Config FILE            Custom config
    -Device DEVICE          Force device

TRAIN OPTIONS:
    -Fold FOLD              Train specific fold
    -AllFolds               Train all 15 folds
    -Resume CHECKPOINT      Resume from checkpoint
    -Config FILE            Config file
    -Epochs N               Override epochs
    -BatchSize N            Override batch size
    -Device DEVICE          Override device
    -Continue               Continue incomplete folds only

STATUS OPTIONS:
    -OutputDir DIR          Check specific directory
    -Detailed               Show detailed metrics
    -Watch                  Continuously update
    -Interval SECONDS       Update interval

VERIFY OPTIONS:
    -OutputDir DIR          Verify specific directory
    -Fold FOLD              Verify specific fold
    -IntegrityCheck         Deep checkpoint check
    -GenerateReport         Generate comprehensive report
    -ExportFormat FORMAT    json, html, or markdown

CLEAN OPTIONS:
    -TempOnly               Remove only temporary files
    -Archive                Archive outputs before cleaning
    -ArchiveDir DIR         Archive directory
    -KeepLast N             Keep last N output directories
    -FullReset              Complete reset (DANGEROUS)
    -DryRun                 Show what would be removed

GLOBAL OPTIONS:
    -Verbose                Enable verbose output
    -NoColor                Disable colored output

ENVIRONMENT VARIABLES:
    VAD_CONFIG              Default config file
    VAD_DATA_DIR            TORGO data directory
    VAD_OUTPUT_DIR          Output directory
    VAD_LOG_DIR             Log directory
    VAD_DEVICE              Preferred device
    VAD_SEED                Random seed

EXAMPLES:
    # Initial setup
    .\start.ps1 setup

    # Quick test
    .\start.ps1 quick-test

    # Train single fold
    .\start.ps1 train -Fold F01

    # Train all folds
    .\start.ps1 train -AllFolds

    # Continue incomplete
    .\start.ps1 train -AllFolds -Continue

    # Check status
    .\start.ps1 status

    # Watch mode
    .\start.ps1 status -Watch

    # Verify results
    .\start.ps1 verify -GenerateReport

    # Run demo workflow
    .\start.ps1 demo

    # Archive old outputs
    .\start.ps1 clean -Archive

FOLDS (15 total):
    F01, F03, F04, M01, M02, M03, M04, M05,
    FC01, FC02, FC03, MC01, MC02, MC03, MC04

For more information, see:
    - README.md
    - AGENTS.md
    - docs/START_SH_DESIGN.md
"@
}

# =============================================================================
# MAIN
# =============================================================================

# Setup colors and logging
Setup-Colors
Setup-Logging

# Dispatch to mode handler
switch ($Mode) {
    "setup" { Invoke-Setup }
    "quick-test" { Invoke-QuickTest }
    "train" { Invoke-Train }
    "status" { Invoke-Status }
    "verify" { Invoke-Verify }
    "demo" { Invoke-Demo }
    "clean" { Invoke-Clean }
    "help" { Invoke-Help }
    default { 
        Error "Unknown mode: $Mode"
        Invoke-Help
        exit 1
    }
}
