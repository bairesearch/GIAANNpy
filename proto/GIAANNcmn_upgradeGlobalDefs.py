"""Parameters of the observed-source, causal routing and shared-input branch upgrades."""

upgradeZero = 0
upgradeOne = 1
upgradeTwo = 2
upgradeAbsent = -1
upgradeSourceWidth = 64
upgradeTraceWindow = 128
upgradeTraceStrength = 0.25
upgradeAdvantageDecay = 0.97
upgradeAdvantageScale = 2.0
upgradeAdvantageLimit = 6.0
upgradeBranchCount = 4
upgradeBranchSlots = 4
upgradeBranchAfferentCache = 128
upgradeBranchRecruitment = 8
upgradeBranchQueryBatch = 512
upgradeBranchReadExposures = "SELECT target,count FROM exposures WHERE target IN ({})"
upgradeBranchReadContacts = "SELECT target,branch,compartment,source,weight FROM contacts WHERE target IN ({})"
upgradeBranchAddExposures = "INSERT INTO exposures VALUES(?,?) ON CONFLICT(target) DO UPDATE SET count=count+excluded.count"
upgradeBranchAddContacts = "INSERT INTO contacts VALUES(?,?,?,?,?) ON CONFLICT(target,branch,compartment,source) DO UPDATE SET weight=weight+excluded.weight"
upgradeProximal = 0
upgradeDistal = 1
upgradeBranchHalf = 0.5
upgradeBranchSmoothing = 8.0
upgradeBranchGainCap = 3.0
upgradeBranchStorageName = "sharedInputDendriticBranches.sqlite3"
upgradeBranchStorageVersion = 1
upgradeUTF8 = "utf-8"
upgradeNeutralPOS = "X"
upgradeInvalidConfiguration = "Prediction upgrades require the fixed aligned subword vocabulary and a single-branch native likelihood readout."
upgradeInvalidSource = "Prediction upgrade received an invalid physical source, candidate, or sequence position."
upgradeInvalidScores = "Prediction upgrade received invalid native or distributed candidate weights."
upgradeInvalidBranchStore = "Shared-input dendritic checkpoint is missing or incompatible. Train with inferenceUpgradeLearnSharedInputDendriticBranches enabled before scoring with branch evidence."
upgradeReadoutAttribute = "upgradeReadout"
upgradePooledCandidatesAttribute = "upgradePooledCandidates"
upgradePooledCandidatesCache = 128
upgradeNoContinuation = "No trained local, observed-source or branch evidence for this prediction"
upgradeInvalidPrefixConfiguration = "Prefix-only evaluation requires teacher-forced native subword likelihood inference, a nonempty seed and fixed first-noun columns."
