& {
    param(
        [Alias("h")][switch]$Help,
        [string]$Config,
        [string]$Version,
        [string]$Branch,
        [string]$Python = "3.13"
    )
    $ErrorActionPreference = "Stop"
    function Fail([string]$Message) { throw "URSA installer: $Message" }
    function Find-Uv {
        $command = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
        if ($command) { return $command.Path }
        foreach ($directory in @($env:UV_UNMANAGED_INSTALL, $env:UV_INSTALL_DIR, "$HOME\.local\bin", "$HOME\.cargo\bin")) {
            if ($directory -and (Test-Path -LiteralPath "$directory\uv.exe" -PathType Leaf)) { return "$directory\uv.exe" }
        }
    }
    if ($Help) { Write-Output "Usage: install.ps1 [-Config FILE] [-Branch BRANCH | -Version VERSION] [-Python VERSION]"; return }

    if ($PSBoundParameters.ContainsKey("Branch") -and $PSBoundParameters.ContainsKey("Version")) { Fail "-Branch and -Version cannot be used together" }
    $ursaRequirement = "ursa-ai"
    if ($PSBoundParameters.ContainsKey("Branch")) { $ursaRequirement = "ursa-ai @ git+https://github.com/lanl/ursa.git@$Branch" }
    if ($PSBoundParameters.ContainsKey("Version")) { $ursaRequirement = "ursa-ai==$Version" }
    $interactive = [Environment]::UserInteractive -and -not [Console]::IsInputRedirected
    if ($Config -and -not (Test-Path -LiteralPath $Config -PathType Leaf)) { Fail "setup defaults not found: $Config" }
    if ($Config -and -not $interactive) { Fail "setup defaults require an interactive terminal" }

    $uv = Find-Uv
    if (-not $uv) {
        Write-Output "Installing uv..."
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        Invoke-RestMethod "https://astral.sh/uv/install.ps1" | Invoke-Expression
        $uv = Find-Uv
        if (-not $uv) { Fail "uv installed, but its executable could not be found" }
    }
    else { Write-Output "Using uv at $uv" }

    Write-Output "Installing URSA with Python $Python..."
    & $uv tool install --python $Python --force --upgrade --compile-bytecode $ursaRequirement
    if ($LASTEXITCODE) { Fail "uv failed to install URSA (exit code $LASTEXITCODE)" }
    $toolBin = "$(& $uv tool dir --bin | Select-Object -Last 1)".Trim()
    $ursa = Join-Path $toolBin "ursa.exe"
    & $uv tool update-shell
    if ($LASTEXITCODE) { Write-Output "Could not update this shell automatically; add $toolBin to PATH." }
    $env:Path = "$toolBin$([IO.Path]::PathSeparator)$(Split-Path -Parent $uv)$([IO.Path]::PathSeparator)$($env:Path)"

    & $ursa self setup --help *> $null
    if ($LASTEXITCODE) {
        $ursaVersion = (& $ursa --version 2> $null | Select-Object -First 1)
        Fail "installed URSA ($ursaVersion) does not support guided setup. See https://lanl.github.io/ursa/latest/getting-started."
    }

    Write-Output "Starting guided setup..."
    $setupArguments = @("self", "setup")
    if ($Config) { $setupArguments += @("--defaults-json", $Config) }
    if ($interactive) {
        & $ursa @setupArguments
        if ($LASTEXITCODE) { Fail "guided setup failed (exit code $LASTEXITCODE)" }
    }
    else {
        Write-Output "No interactive terminal was found. Finish setup by running:"
        Write-Output "  ursa self setup"
    }
    Write-Output "URSA is installed. Run 'ursa' to get started."
} @args
