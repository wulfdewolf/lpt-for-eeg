library(nortest)
library(car)

# Read results
results <- read.csv(file = "./scripts/analysis/export_test.csv", stringsAsFactors = FALSE)

# Two means
group1 <- results[results[["run_type"]]=="3FONCO","Test.Accuracy"]
group2 <- results[results[["run_type"]]!="3FONCO","Test.Accuracy"]
lillie.test(group1-group2)
t.test(group1, group2, paired=TRUE) #, alternative="greater")

# Multiple means
group1 <- results[results[["run_type"]]=="Z19MQT","Test.Accuracy"]
lillie.test(group1)
group2 <- results[results[["run_type"]]=="WONM6L","Test.Accuracy"]
lillie.test(group2)
group3 <- results[results[["run_type"]]=="935F7M","Test.Accuracy"]
lillie.test(group3)
leveneTest(Test.Accuracy~run_type, results)
summary(aov(Test.Accuracy~run_type, data=results))
