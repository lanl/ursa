& {
    [CmdletBinding()]
    param(
        [Alias("h")]
        [switch]$Help,

        [string]$Config,

        [string]$Version,

        [string]$Branch,

        [string]$Python = "3.13"
    )

    Set-StrictMode -Version Latest
    $ErrorActionPreference = "Stop"

    function Stop-Installer {
        param([string]$Message)
        throw "URSA installer: $Message"
    }

    function Find-UvExecutable {
        $uvCommand = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -ne $uvCommand) {
            return $uvCommand.Path
        }

        foreach ($directory in @(
            $env:UV_UNMANAGED_INSTALL, $env:UV_INSTALL_DIR,
            "$HOME\.local\bin", "$HOME\.cargo\bin"
        )) {
            if (-not $directory) { continue }
            $candidate = Join-Path $directory "uv.exe"
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                return (Get-Item -LiteralPath $candidate -Force).FullName
            }
        }
    }

    function Test-InteractiveTerminal {
        try { return [Environment]::UserInteractive -and -not [Console]::IsInputRedirected }
        catch { return $Host.Name -eq "ConsoleHost" }
    }

    if ($PSVersionTable.PSVersion -lt [Version]"5.1") {
        Stop-Installer "PowerShell 5.1 or newer is required"
    }

    if ($Help) {
        @"
Install URSA with uv and start the guided setup.

Usage: install.ps1 [-Config FILE] [-Branch BRANCH | -Version VERSION] [-Python VERSION]

  -Config FILE              Read setup defaults from a JSON file.
  -Branch BRANCH            Install from a branch in the LANL URSA repository.
  -Version VERSION          Install a PyPI version.
  -Python VERSION           Install with this Python version (default: 3.13).
"@ | Write-Output
        return
    }

    $ursaRequirement = "ursa-ai"
    if ($PSBoundParameters.ContainsKey("Version")) {
        $ursaRequirement = "ursa-ai==$Version"
    }
    if ($PSBoundParameters.ContainsKey("Branch")) {
        $ursaRequirement = "ursa-ai @ git+https://github.com/lanl/ursa.git@$Branch"
    }
    try {
        if ($PSBoundParameters.ContainsKey("Config")) {
            if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) {
                Stop-Installer "setup defaults not found: $Config"
            }
            if (-not (Test-InteractiveTerminal)) {
                Stop-Installer "setup defaults require an interactive terminal"
            }
        }

        $uvExecutable = Find-UvExecutable
        if (-not $uvExecutable) {
            Write-Output "Installing uv..."

            [Net.ServicePointManager]::SecurityProtocol =
                [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
            Invoke-RestMethod "https://astral.sh/uv/install.ps1" | Invoke-Expression

            $uvExecutable = Find-UvExecutable
            if (-not $uvExecutable) {
                Stop-Installer "uv installed, but its executable could not be found"
            }
        }
        else {
            Write-Output "Using uv at $uvExecutable"
        }

        Write-Output "Installing URSA with Python $Python..."
        & $uvExecutable tool install --python $Python --force --upgrade --compile-bytecode $ursaRequirement
        if ($LASTEXITCODE -ne 0) {
            Stop-Installer "uv failed to install URSA (exit code $LASTEXITCODE)"
        }

        $toolBin = (& $uvExecutable tool dir --bin | Select-Object -Last 1)
        if ($LASTEXITCODE -ne 0 -or -not $toolBin) {
            Stop-Installer "uv returned an empty tool bin directory"
        }
        $toolBin = "$toolBin".Trim()

        $ursaExecutable = Join-Path $toolBin "ursa.exe"
        if (-not (Test-Path -LiteralPath $ursaExecutable -PathType Leaf)) {
            Stop-Installer "URSA installed, but its executable was not found at $ursaExecutable"
        }
        $ursaExecutable = (Get-Item -LiteralPath $ursaExecutable -Force).FullName

        $currentUrsaCommand = Get-Command ursa -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        $currentUrsa = if ($currentUrsaCommand) { $currentUrsaCommand.Path }
        $ursaCommand = "ursa"
        $ursaIsCurrent = $currentUrsa -and $currentUrsa.Equals(
            $ursaExecutable, [StringComparison]::OrdinalIgnoreCase
        )
        if ($currentUrsa -and -not $ursaIsCurrent) {
            Write-Output "Warning: $currentUrsa currently takes precedence over $ursaExecutable."
            $ursaCommand = $ursaExecutable
        }

        if (-not $ursaIsCurrent) {
            Write-Output "Adding uv's tool directory to PATH..."
            & $uvExecutable tool update-shell
            if ($LASTEXITCODE -ne 0) {
                Write-Output "Could not update this shell automatically; add $toolBin to PATH."
            }
            Write-Output "Follow uv's instructions above to update this shell. Or restart your shell."
        }

        $pathSeparator = [IO.Path]::PathSeparator
        $uvDirectory = Split-Path -Parent $uvExecutable
        $env:Path = "$toolBin$pathSeparator$uvDirectory$pathSeparator$($env:Path)"

        & $ursaExecutable self setup --help *> $null
        if ($LASTEXITCODE -ne 0) {
            $ursaVersion = (& $ursaExecutable --version 2> $null | Select-Object -First 1)
            if ($LASTEXITCODE -ne 0 -or -not $ursaVersion) {
                $ursaVersion = "version unavailable"
            }
            Stop-Installer "installed URSA ($ursaVersion) does not support guided setup. See https://lanl.github.io/ursa/latest/getting-started."
        }

        Write-Output "Starting guided setup..."
        $setupArguments = @("self", "setup")
        if ($Config) {
            $setupArguments += @("--defaults-json", $Config)
        }
        if (Test-InteractiveTerminal) {
            & $ursaExecutable @setupArguments
            if ($LASTEXITCODE -ne 0) {
                Stop-Installer "guided setup failed (exit code $LASTEXITCODE)"
            }
        }
        else {
            $setupCommand = if ($ursaCommand -eq "ursa") {
                "ursa self setup"
            }
            else {
                "& `"$ursaCommand`" self setup"
            }
            Write-Output "No interactive terminal was found. Finish setup by running:"
            Write-Output "  $setupCommand"
        }

        $commandDisplay = if ($ursaCommand -eq "ursa") { "ursa" } else { "& `"$ursaCommand`"" }
        Write-Output "URSA is installed. Run '$commandDisplay' to get started."
    }
    catch {
        throw "URSA installer: $($_.Exception.Message -replace '^URSA installer: ', '')"
    }
} @args
