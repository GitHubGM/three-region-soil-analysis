[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$SourceDocx,
    [Parameter(Mandatory = $true)][string]$OutputPdf
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$workspaceDirectory = [System.IO.Path]::GetFullPath('C:\proj\papers\ERA-5')

function Get-WorkspacePath {
    param([string]$Candidate, [string]$RequiredExtension, [bool]$MustExist)
    $fullPath = [System.IO.Path]::GetFullPath($Candidate)
    $workspacePrefix = $workspaceDirectory.TrimEnd('\') + '\'
    if (-not $fullPath.StartsWith($workspacePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path must be inside $workspaceDirectory : $Candidate"
    }
    if ([System.IO.Path]::GetExtension($fullPath) -ine $RequiredExtension) {
        throw "Expected a $RequiredExtension path: $Candidate"
    }
    if ($MustExist -and -not [System.IO.File]::Exists($fullPath)) {
        throw "Input file does not exist: $fullPath"
    }
    # Reject junctions and symlinks so a lexically confined path cannot escape.
    $ancestorPath = $fullPath
    while ($ancestorPath -and $ancestorPath.Length -ge $workspaceDirectory.Length) {
        if (Test-Path -LiteralPath $ancestorPath) {
            $entry = Get-Item -LiteralPath $ancestorPath -Force
            if (($entry.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Reparse points are not allowed for this export: $ancestorPath"
            }
        }
        $ancestorPath = [System.IO.Path]::GetDirectoryName($ancestorPath)
    }
    return $fullPath
}

$sourcePath = Get-WorkspacePath $SourceDocx '.docx' $true
$pdfPath = Get-WorkspacePath $OutputPdf '.pdf' $false
$outputDirectory = [System.IO.Path]::GetDirectoryName($pdfPath)
[System.IO.Directory]::CreateDirectory($outputDirectory) | Out-Null

$ownedWord = $null
$ownedDocument = $null
try {
    $ownedWord = New-Object -ComObject Word.Application
    $ownedWord.Visible = $false
    $ownedWord.DisplayAlerts = 0
    $ownedWord.AutomationSecurity = 3
    # ConfirmConversions false, ReadOnly true, AddToRecentFiles false.
    $ownedDocument = $ownedWord.Documents.Open($sourcePath, $false, $true, $false)
    $ownedDocument.Repaginate()
    $pageCount = $ownedDocument.ComputeStatistics(2)
    $ownedDocument.ExportAsFixedFormat($pdfPath, 17)
    if (-not [System.IO.File]::Exists($pdfPath) -or (Get-Item -LiteralPath $pdfPath).Length -eq 0) {
        throw "Word did not produce a nonempty PDF: $pdfPath"
    }
    [pscustomobject]@{
        source = $sourcePath
        pdf = $pdfPath
        pages = $pageCount
        renderer = 'Microsoft Word'
    } | ConvertTo-Json -Compress
}
finally {
    # Release only the document and application created by this script.
    if ($null -ne $ownedDocument) {
        try { $ownedDocument.Close(0) }
        catch { Write-Warning "Could not close the exported document: $_" }
        try { [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($ownedDocument) | Out-Null }
        catch { Write-Warning "Could not release the document COM object: $_" }
    }
    if ($null -ne $ownedWord) {
        try { $ownedWord.Quit(0) }
        catch { Write-Warning "Could not close the owned Word instance: $_" }
        try { [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($ownedWord) | Out-Null }
        catch { Write-Warning "Could not release the Word COM object: $_" }
    }
}
