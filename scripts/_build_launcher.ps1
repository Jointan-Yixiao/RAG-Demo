<#
Build the Windows launcher RAG-Workbench.exe with the .NET Framework compiler that ships with Windows.

  powershell -NoProfile -File scripts\_build_launcher.ps1          # build RAG-Workbench.exe at project root
  powershell -NoProfile -File scripts\_build_launcher.ps1 -Test    # also build and run offline tests

No packages are downloaded, no administrator rights are needed, and no machine/user policy is changed.
If local policy blocks script execution, use the included executable or ask your administrator.
This script never changes execution policy.

Outputs (all inside the project):
  RAG-Workbench.exe                              launcher (winexe, AnyCPU, icon assets\app.ico)
  data\ui\launcher-tests\LauncherTests.exe       console test build (-Test only)
  data\ui\launcher-tests\run-<timestamp>\        per-run temporary roots and shortcuts (-Test only; not deleted)
#>
[CmdletBinding()]
param(
    [switch]$Test
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

function Find-Csc {
    $candidates = @(
        (Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'),
        (Join-Path $env:WINDIR 'Microsoft.NET\Framework\v4.0.30319\csc.exe')
    )
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c -PathType Leaf) { return $c }
    }
    throw 'The .NET Framework 4 compiler (csc.exe) was not found under %WINDIR%\Microsoft.NET. Enable .NET Framework 4.8 in Windows Features.'
}

function Invoke-Csc {
    param([string]$Csc, [string[]]$Arguments)
    # Arguments are passed as an array; PowerShell quotes each element. No cmd /c.
    & $Csc @Arguments
    if ($LASTEXITCODE -ne 0) { throw "csc.exe failed with exit code $LASTEXITCODE" }
}

$csc = Find-Csc
$frameworkDir = Split-Path -Parent $csc
$source = Join-Path $root 'launcher\WorkbenchLauncher.cs'
$icon = Join-Path $root 'assets\app.ico'
$out = Join-Path $root 'RAG-Workbench.exe'

foreach ($required in @($source, $icon)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing required file: $required" }
}

$references = @('System.dll', 'System.Drawing.dll', 'System.Windows.Forms.dll', 'System.Web.Extensions.dll') |
    ForEach-Object { '/reference:' + (Join-Path $frameworkDir $_) }

$common = @('/nologo', '/codepage:65001', '/utf8output', '/optimize+', '/warn:4', '/platform:anycpu') + $references

Write-Host "Compiler: $csc"
Write-Host "Building $out"
Invoke-Csc $csc ($common + @('/target:winexe', "/win32icon:$icon", "/out:$out", $source))
Write-Host 'Launcher built.'

if ($Test) {
    $testDir = Join-Path $root 'data\ui\launcher-tests'
    New-Item -ItemType Directory -Force -Path $testDir | Out-Null
    $testSource = Join-Path $root 'tests\launcher\LauncherTests.cs'
    $testExe = Join-Path $testDir 'LauncherTests.exe'
    if (-not (Test-Path -LiteralPath $testSource -PathType Leaf)) { throw "Missing test source: $testSource" }

    Write-Host "Building $testExe"
    Invoke-Csc $csc ($common + @('/target:exe', '/main:RagWorkbench.Tests.LauncherTests', "/out:$testExe", $source, $testSource))

    Write-Host 'Running launcher tests'
    & $testExe $testDir
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "Launcher tests failed with exit code $code" }
    Write-Host 'Launcher tests passed.'
}
