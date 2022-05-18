library(nortest)

# Read results
results <- read.csv(file = "./scripts/analysis/export_test.csv", stringsAsFactors = FALSE)
group1 <- results[results[["hyperparams.decay"]]!=1,"Test.Accuracy"]
group2 <- results[results[["hyperparams.decay"]]==1,"Test.Accuracy"]

# Test normality
lillie.test(group1)
lillie.test(group2)

# Test means
t.test(group1, group2, paired=TRUE, alternative="less")
