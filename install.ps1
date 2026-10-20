& {
    [CmdletBinding()]
    param(
        [Alias("h")]
        [switch]$Help,

        [string]$Config,

        [string]$ConfigJson
    )

    Set-StrictMode -Version Latest
    $ErrorActionPreference = "Stop"

    function Write-InstallerMessage {
        param([Parameter(Mandatory = $true)][string]$Message)

        Write-Output $Message
    }

    function Stop-Installer {
        param([Parameter(Mandatory = $true)][string]$Message)

        throw "URSA installer: $Message"
    }

    function Find-UvExecutable {
        $uvCommand = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -ne $uvCommand) {
            return $uvCommand.Path
        }

        $candidates = @()
        if ($env:UV_UNMANAGED_INSTALL) {
            $candidates += Join-Path $env:UV_UNMANAGED_INSTALL "uv.exe"
        }
        if ($env:UV_INSTALL_DIR) {
            $candidates += Join-Path $env:UV_INSTALL_DIR "uv.exe"
        }
        if ($env:XDG_BIN_HOME) {
            $candidates += Join-Path $env:XDG_BIN_HOME "uv.exe"
        }
        if ($env:XDG_DATA_HOME) {
            $candidates += Join-Path $env:XDG_DATA_HOME "..\bin\uv.exe"
        }
        if ($HOME) {
            $candidates += Join-Path $HOME ".local\bin\uv.exe"
            $candidates += Join-Path $HOME ".cargo\bin\uv.exe"
        }
        if ($env:LOCALAPPDATA) {
            $candidates += Join-Path $env:LOCALAPPDATA "uv\bin\uv.exe"
            $candidates += Join-Path $env:LOCALAPPDATA "Programs\uv\uv.exe"
        }

        foreach ($candidate in $candidates) {
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                return (Get-Item -LiteralPath $candidate -Force).FullName
            }
        }

        return $null
    }

    function Test-InteractiveTerminal {
        if (-not [Environment]::UserInteractive) {
            return $false
        }

        try {
            return -not [Console]::IsInputRedirected
        }
        catch {
            return $Host.Name -eq "ConsoleHost"
        }
    }

    if ($PSVersionTable.PSVersion -lt [Version]"5.1") {
        Stop-Installer "PowerShell 5.1 or newer is required"
    }

    if ($Help) {
        @"
Install URSA with uv and start the guided setup.

Usage: install.ps1 [-Config FILE | -ConfigJson JSON]

  -Config FILE       Read a setup defaults JSON payload from FILE.
  -ConfigJson JSON   Read the payload directly from the argument.
"@ | Write-Output
        return
    }

    $configSourceCount = @(
        $PSBoundParameters.ContainsKey("Config")
        $PSBoundParameters.ContainsKey("ConfigJson")
    ).Where({ $_ }).Count
    if ($configSourceCount -gt 1) {
        Stop-Installer "setup defaults may only be specified once"
    }

    $payloadFile = $null
    try {
        if ($configSourceCount -eq 1) {
            $payloadFile = [IO.Path]::GetTempFileName()
            if ($PSBoundParameters.ContainsKey("Config")) {
                if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) {
                    Stop-Installer "setup defaults not found: $Config"
                }
                Copy-Item -LiteralPath $Config -Destination $payloadFile -Force
            }
            else {
                $utf8 = New-Object Text.UTF8Encoding($false)
                [IO.File]::WriteAllText($payloadFile, $ConfigJson, $utf8)
            }
            if (-not (Test-InteractiveTerminal)) {
                Stop-Installer "setup defaults require an interactive terminal"
            }
        }

        $uvExecutable = Find-UvExecutable
        if (-not $uvExecutable) {
            Write-InstallerMessage "Installing uv..."

            [Net.ServicePointManager]::SecurityProtocol =
                [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
            $installerWork = Join-Path ([IO.Path]::GetTempPath()) (
                "ursa-installer-" + [Guid]::NewGuid().ToString("N")
            )
            New-Item -ItemType Directory -Path $installerWork | Out-Null
            try {
                $uvInstaller = Join-Path $installerWork "uv-install.ps1"
                Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" `
                    -UseBasicParsing -OutFile $uvInstaller
                & $uvInstaller
            }
            finally {
                Remove-Item -LiteralPath $installerWork -Recurse -Force `
                    -ErrorAction SilentlyContinue
            }

            $uvExecutable = Find-UvExecutable
            if (-not $uvExecutable) {
                Stop-Installer "uv installed, but its executable could not be found"
            }
        }
        else {
            Write-InstallerMessage "Using uv at $uvExecutable"
        }

        Write-InstallerMessage "Installing URSA with Python 3.13..."
        & $uvExecutable tool install --python 3.13 --force --upgrade --compile-bytecode ursa-ai
        $uvInstallExitCode = $LASTEXITCODE
        if ($uvInstallExitCode -ne 0) {
            Stop-Installer "uv failed to install URSA (exit code $uvInstallExitCode)"
        }

        $toolBinOutput = @(& $uvExecutable tool dir --bin)
        $toolBinExitCode = $LASTEXITCODE
        if ($toolBinExitCode -ne 0) {
            Stop-Installer "uv could not locate its tool bin directory (exit code $toolBinExitCode)"
        }
        $toolBinLines = @($toolBinOutput | ForEach-Object { "$($_)".Trim() } | Where-Object { $_ })
        if ($toolBinLines.Count -eq 0) {
            Stop-Installer "uv returned an empty tool bin directory"
        }
        $toolBinValue = $toolBinLines[$toolBinLines.Count - 1]
        if (-not (Test-Path -LiteralPath $toolBinValue -PathType Container)) {
            Stop-Installer "uv returned an invalid tool bin directory: $toolBinValue"
        }
        $toolBin = (Get-Item -LiteralPath $toolBinValue -Force).FullName

        $ursaExecutable = Join-Path $toolBin "ursa.exe"
        if (-not (Test-Path -LiteralPath $ursaExecutable -PathType Leaf)) {
            Stop-Installer "URSA installed, but its executable was not found at $ursaExecutable"
        }
        $ursaExecutable = (Get-Item -LiteralPath $ursaExecutable -Force).FullName
        $uvExecutable = (Get-Item -LiteralPath $uvExecutable -Force).FullName
        $uvDirectory = Split-Path -Parent $uvExecutable

        $currentUrsaCommand = Get-Command ursa -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        $currentUrsa = if ($null -ne $currentUrsaCommand) { $currentUrsaCommand.Path } else { $null }
        $ursaCommand = "ursa"
        if ($currentUrsa -and -not $currentUrsa.Equals($ursaExecutable, [StringComparison]::OrdinalIgnoreCase)) {
            Write-InstallerMessage "Warning: $currentUrsa currently takes precedence over $ursaExecutable."
            $ursaCommand = $ursaExecutable
        }

        if (-not $currentUrsa -or -not $currentUrsa.Equals($ursaExecutable, [StringComparison]::OrdinalIgnoreCase)) {
            Write-InstallerMessage "Adding uv's tool directory to PATH..."
            & $uvExecutable tool update-shell
            $updateShellExitCode = $LASTEXITCODE
            if ($updateShellExitCode -ne 0) {
                Write-InstallerMessage "Could not update this shell automatically; add $toolBin to PATH."
            }
            Write-InstallerMessage "Follow uv's instructions above to update this shell. Or restart your shell."
        }

        $pathSeparator = [IO.Path]::PathSeparator
        $env:Path = "$toolBin$pathSeparator$uvDirectory$pathSeparator$($env:Path)"

        & $ursaExecutable self setup --help *> $null
        $setupHelpExitCode = $LASTEXITCODE
        if ($setupHelpExitCode -ne 0) {
            $ursaVersionOutput = @(& $ursaExecutable --version 2> $null)
            $ursaVersionExitCode = $LASTEXITCODE
            $ursaVersion = ($ursaVersionOutput | ForEach-Object { "$($_)".Trim() } | Where-Object { $_ } |
                Select-Object -First 1)
            if ($ursaVersionExitCode -ne 0 -or -not $ursaVersion) {
                $ursaVersion = "version unavailable"
            }
            Stop-Installer "installed URSA ($ursaVersion) does not support guided setup. See https://lanl.github.io/ursa/latest/getting-started."
        }

        Write-InstallerMessage "Starting guided setup..."
        $setupArguments = @("self", "setup")
        if ($payloadFile) {
            $setupArguments += @("--defaults-json", $payloadFile)
        }
        if (Test-InteractiveTerminal) {
            & $ursaExecutable @setupArguments
            $setupExitCode = $LASTEXITCODE
            if ($setupExitCode -ne 0) {
                Stop-Installer "guided setup failed (exit code $setupExitCode)"
            }
        }
        else {
            $setupCommand = if ($ursaCommand -eq "ursa") {
                "ursa self setup"
            }
            else {
                "& `"$ursaCommand`" self setup"
            }
            Write-InstallerMessage "No interactive terminal was found. Finish setup by running:"
            Write-InstallerMessage "  $setupCommand"
        }

        $commandDisplay = if ($ursaCommand -eq "ursa") { "ursa" } else { "& `"$ursaCommand`"" }
        Write-InstallerMessage "URSA is installed. Run '$commandDisplay' to get started."
    }
    catch {
        if ($_.Exception.Message.StartsWith("URSA installer: ", [StringComparison]::Ordinal)) {
            throw
        }
        throw "URSA installer: $($_.Exception.Message)"
    }
    finally {
        if ($payloadFile) {
            Remove-Item -LiteralPath $payloadFile -Force -ErrorAction SilentlyContinue
        }
    }
} @args
