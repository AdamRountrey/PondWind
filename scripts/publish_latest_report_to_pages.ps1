param(
    [string]$ReportDir = "",
    [string]$ReportsRoot = "",
    [string]$WorktreePath = "",
    [string]$Branch = "gh-pages",
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [switch]$AllowFailure
    )
    & git @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0 -and -not $AllowFailure) {
        throw "git $($Arguments -join ' ') failed with exit code $exitCode"
    }
    return $exitCode
}

function Test-GitRef {
    param([string]$Ref)
    & git -C $projectRoot rev-parse --verify --quiet $Ref *> $null
    return $LASTEXITCODE -eq 0
}

function ConvertTo-HtmlText {
    param([object]$Value)
    if ($null -eq $Value) {
        return ""
    }
    return [System.Net.WebUtility]::HtmlEncode([string]$Value)
}

function ConvertTo-SafeSiteLabel {
    param(
        [object]$Manifest,
        [System.IO.DirectoryInfo]$ReportDirectory
    )
    $candidate = $null
    if ($null -ne $Manifest -and $null -ne $Manifest.site -and $null -ne $Manifest.site.label) {
        $candidate = [string]$Manifest.site.label
    }
    if ([string]::IsNullOrWhiteSpace($candidate)) {
        $candidate = $ReportDirectory.Name
    }
    $candidate = $candidate.Trim()
    if ($candidate -match '^[A-Za-z0-9_-]{1,64}$') {
        return $candidate
    }
    return "pondwind"
}

function Resolve-ReportDirectory {
    if (![string]::IsNullOrWhiteSpace($ReportDir)) {
        if (!(Test-Path -LiteralPath $ReportDir -PathType Container)) {
            throw "ReportDir was not found: $ReportDir"
        }
        return Get-Item -LiteralPath $ReportDir
    }

    if ([string]::IsNullOrWhiteSpace($ReportsRoot)) {
        $localAppData = $env:LOCALAPPDATA
        if ([string]::IsNullOrWhiteSpace($localAppData)) {
            throw "LOCALAPPDATA is not set. Pass -ReportsRoot or -ReportDir."
        }
        $ReportsRoot = Join-Path $localAppData "PondWind\outputs\reports"
    }
    if (!(Test-Path -LiteralPath $ReportsRoot -PathType Container)) {
        throw "ReportsRoot was not found: $ReportsRoot"
    }
    $latest = Get-ChildItem -LiteralPath $ReportsRoot -Directory |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latest) {
        throw "No report folders were found under $ReportsRoot"
    }
    return $latest
}

function Ensure-PagesWorktree {
    param([string]$TargetPath)

    if ([string]::IsNullOrWhiteSpace($TargetPath)) {
        $TargetPath = Join-Path $projectRoot ".worktrees\gh-pages"
    }
    $parent = Split-Path -Parent $TargetPath
    if (!(Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent | Out-Null
    }

    $remoteBranchRef = "refs/remotes/origin/$Branch"
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & git -C $projectRoot fetch origin $Branch *> $null
        $null = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if (Test-Path -LiteralPath $TargetPath) {
        if (!(Test-Path -LiteralPath (Join-Path $TargetPath ".git"))) {
            throw "WorktreePath exists but is not a Git worktree: $TargetPath"
        }
        Invoke-Git -Arguments @("-C", $TargetPath, "checkout", $Branch) | Out-Null
    } elseif (Test-GitRef $Branch) {
        Invoke-Git -Arguments @("-C", $projectRoot, "worktree", "add", $TargetPath, $Branch) | Out-Null
    } elseif (Test-GitRef $remoteBranchRef) {
        Invoke-Git -Arguments @("-C", $projectRoot, "worktree", "add", "-B", $Branch, $TargetPath, "origin/$Branch") | Out-Null
    } else {
        Invoke-Git -Arguments @("-C", $projectRoot, "worktree", "add", "--detach", $TargetPath, "HEAD") | Out-Null
        Invoke-Git -Arguments @("-C", $TargetPath, "switch", "--orphan", $Branch) | Out-Null
        Invoke-Git -Arguments @("-C", $TargetPath, "rm", "-r", "--ignore-unmatch", ".") -AllowFailure | Out-Null
        Get-ChildItem -LiteralPath $TargetPath -Force |
            Where-Object { $_.Name -ne ".git" } |
            ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }
    }

    if (Test-GitRef $remoteBranchRef) {
        Invoke-Git -Arguments @("-C", $TargetPath, "pull", "--ff-only", "origin", $Branch) -AllowFailure | Out-Null
    }

    $dirty = & git -C $TargetPath status --porcelain
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect Pages worktree status."
    }
    if ($dirty) {
        throw "Pages worktree has uncommitted changes. Clean it first: $TargetPath"
    }
    return (Resolve-Path $TargetPath).Path
}

function Add-ProductIfPresent {
    param(
        [System.Collections.ArrayList]$Products,
        [System.IO.DirectoryInfo]$ReportDirectory,
        [string]$FileName,
        [string]$Title,
        [string]$Category
    )
    $source = Join-Path $ReportDirectory.FullName $FileName
    if (!(Test-Path -LiteralPath $source -PathType Leaf)) {
        return
    }
    $Products.Add([ordered]@{
        title = $Title
        category = $Category
        image = "latest/images/$FileName"
        file_name = $FileName
    }) | Out-Null
}

$reportDirectory = Resolve-ReportDirectory
$manifestPath = Join-Path $reportDirectory.FullName "report_manifest.json"
$manifest = $null
if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
}

$pagesRoot = Ensure-PagesWorktree -TargetPath $WorktreePath
$latestDir = Join-Path $pagesRoot "latest"
$imageDir = Join-Path $latestDir "images"
if (Test-Path -LiteralPath $latestDir) {
    Remove-Item -LiteralPath $latestDir -Recurse -Force
}
New-Item -ItemType Directory -Path $imageDir | Out-Null

$products = [System.Collections.ArrayList]::new()
Add-ProductIfPresent $products $reportDirectory "product_1_wind_speed_prediction_knots.png" "Wind Speed Prediction" "Wind"
Add-ProductIfPresent $products $reportDirectory "product_2_wind_speed_variance_knots.png" "Wind Speed Spread" "Wind"
Add-ProductIfPresent $products $reportDirectory "product_3_wind_direction_variance_degrees.png" "Wind Direction Spread" "Wind"
Add-ProductIfPresent $products $reportDirectory "product_4_openfoam_experimental_cfd_knots.png" "Experimental CFD Comparison" "Experimental CFD"
Add-ProductIfPresent $products $reportDirectory "product_5_openfoam_turbulence_intensity_percent.png" "Experimental CFD Turbulence Intensity" "Experimental CFD"
Add-ProductIfPresent $products $reportDirectory "satellite_rgb_latest.png" "Latest RGB Satellite" "Satellite"
Add-ProductIfPresent $products $reportDirectory "satellite_sst_latest.png" "Surface Temperature" "Satellite"
Add-ProductIfPresent $products $reportDirectory "satellite_chla_estimated.png" "Estimated Chlorophyll-a" "Satellite"
Add-ProductIfPresent $products $reportDirectory "satellite_turbidity_estimated.png" "Estimated Turbidity" "Satellite"

if ($products.Count -eq 0) {
    throw "No whitelisted report images were found in $($reportDirectory.FullName)"
}

foreach ($product in $products) {
    Copy-Item -LiteralPath (Join-Path $reportDirectory.FullName $product.file_name) -Destination (Join-Path $imageDir $product.file_name) -Force
}

$siteLabel = ConvertTo-SafeSiteLabel -Manifest $manifest -ReportDirectory $reportDirectory
$raceTimeLocal = $null
if ($null -ne $manifest -and $null -ne $manifest.race_time_local) {
    $raceTimeLocal = [string]$manifest.race_time_local
}

$reportJson = [ordered]@{
    version = 1
    title = "PondWind Latest Report"
    site_label = $siteLabel
    race_time_local = $raceTimeLocal
    published_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    source_report_folder = $reportDirectory.Name
    products = @($products)
}
$reportJson | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $latestDir "report.json") -Encoding UTF8

$cards = foreach ($product in $products) {
    $title = ConvertTo-HtmlText $product.title
    $category = ConvertTo-HtmlText $product.category
    $image = ConvertTo-HtmlText $product.image
    @"
      <article class="product-card">
        <div class="product-meta">$category</div>
        <h2>$title</h2>
        <a href="$image"><img src="$image" alt="$title"></a>
      </article>
"@
}

$raceText = if ([string]::IsNullOrWhiteSpace($raceTimeLocal)) { "Latest published report" } else { "Race time: $(ConvertTo-HtmlText $raceTimeLocal)" }
$siteText = ConvertTo-HtmlText $siteLabel
$publishedText = ConvertTo-HtmlText $reportJson.published_at_utc
$cardHtml = $cards -join "`n"

$indexHtml = @"
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PondWind Latest Report</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="site-header">
    <p class="eyebrow">PondWind</p>
    <h1>Latest Report</h1>
    <p class="subtitle">$raceText</p>
    <p class="meta">Site: $siteText &middot; Published: $publishedText</p>
  </header>
  <main class="product-grid">
$cardHtml
  </main>
  <footer class="site-footer">
    <p>Generated from sanitized PondWind report images. Raw report manifests and local paths are not published.</p>
  </footer>
</body>
</html>
"@
$indexHtml | Set-Content -LiteralPath (Join-Path $pagesRoot "index.html") -Encoding UTF8

$stylesCss = @"
:root {
  color-scheme: light;
  font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
  background: #f3f6f8;
  color: #172026;
}

body {
  margin: 0;
  background: #f3f6f8;
}

.site-header,
.site-footer {
  max-width: 1180px;
  margin: 0 auto;
  padding: 28px 20px 10px;
}

.eyebrow {
  margin: 0 0 6px;
  color: #27736d;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}

h1 {
  margin: 0;
  font-size: 42px;
  line-height: 1.1;
}

.subtitle,
.meta,
.site-footer {
  color: #52616a;
}

.product-grid {
  max-width: 1180px;
  margin: 18px auto 36px;
  padding: 0 20px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 18px;
}

.product-card {
  background: #ffffff;
  border: 1px solid #d9e2e7;
  border-radius: 8px;
  padding: 14px;
  box-shadow: 0 10px 24px rgba(20, 40, 50, 0.08);
}

.product-card h2 {
  margin: 4px 0 12px;
  font-size: 20px;
  line-height: 1.25;
}

.product-meta {
  color: #27736d;
  font-size: 13px;
  font-weight: 700;
}

.product-card img {
  display: block;
  width: 100%;
  height: auto;
  border: 1px solid #e2e8ec;
}
"@
$stylesCss | Set-Content -LiteralPath (Join-Path $pagesRoot "styles.css") -Encoding UTF8
"" | Set-Content -LiteralPath (Join-Path $pagesRoot ".nojekyll") -Encoding ASCII

Invoke-Git -Arguments @("-C", $pagesRoot, "add", "index.html", "styles.css", ".nojekyll", "latest") | Out-Null
$pending = & git -C $pagesRoot status --porcelain
if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect pending Pages changes."
}
if (-not $pending) {
    Write-Host "No Pages changes to publish."
    exit 0
}

$commitTime = if ([string]::IsNullOrWhiteSpace($raceTimeLocal)) { Get-Date } else { [datetime]::Parse($raceTimeLocal) }
$commitMessage = "Publish latest PondWind report $($commitTime.ToString('yyyy-MM-dd HHmm'))"
Invoke-Git -Arguments @("-C", $pagesRoot, "commit", "-m", $commitMessage) | Out-Null

if ($NoPush) {
    Write-Host "Created local $Branch commit without pushing because -NoPush was set."
} else {
    Invoke-Git -Arguments @("-C", $pagesRoot, "push", "-u", "origin", $Branch) | Out-Null
    Write-Host "Published latest report to $Branch."
}
Write-Host "Pages worktree: $pagesRoot"
Write-Host "Report source: $($reportDirectory.FullName)"
