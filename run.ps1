# Windows PowerShell version of run.sh. Run from the project root in a FRESH COPY of the project:
#   powershell -ExecutionPolicy Bypass -File run.ps1
# It overwrites results\ and figures\. The multi-hour B1 search is listed at the bottom, not run.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
function Run($argList) {
    Write-Host ">> python $argList"
    & python @argList
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: python $argList"; exit 1 }
}
$start = Get-Date

# 0. Tests (made-up data only)
foreach ($t in "test_core","test_my_core","test_lr_check","test_extras","test_h1b","test_b1","test_b2","test_b4","test_report") { Run @("tests/$t.py") }

# 1. Pre-run learning-rate check
Run @("src/lr_check.py")

# 2. Main run
Run @("src/train.py","--widths","2","3","4","6","8","16","32","64","--seeds","40","--steps","3000")
Run @("src/analyze.py","--max-train-err","0.02")
Run @("src/h1b.py")
Run @("src/answers.py")
Run @("src/plot.py","--max-train-err","0.02")
Run @("src/report.py")

# 3. Inclusion sensitivity
Run @("src/analyze.py","--max-train-err","0.01","--out-dir","results_inc1")
Run @("src/analyze.py","--max-train-err","0.05","--out-dir","results_inc5")
Run @("src/answers.py","--results","results_inc1")
Run @("src/answers.py","--results","results_inc5")

# 4. B2 figure
Run @("src/b2_figure.py")

# 5. B4
Run @("src/train.py","--widths","64","--seeds","40","--steps","3000","--wd","0.003","--out","results/models_wd.pkl")
Run @("-c","import pickle; D=pickle.load(open('results/models_wd.pkl','rb')); e=[r['train_err'] for r in D['runs']]; print('wd run: included at 2%:', sum(x<=0.02 for x in e), 'of', len(e))")
Run @("src/analyze.py","--models","results/models_wd.pkl","--max-train-err","1.0","--out-dir","results_wd_all")
Run @("src/b4.py","--wd","results_wd_all","--models-wd","results/models_wd.pkl","--exploratory")

$mins = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
Write-Host "Done in $mins minutes. Optional, several hours: python src/run_b1.py"
