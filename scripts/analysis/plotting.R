library(ggplot2)
library(forcats)

###
# Training/validation score
###

# Read results
results <- read.csv(file = "./scripts/analysis/export_train.csv", stringsAsFactors = FALSE)
results_features <- results[,c(1,2,3,4)]
results_time <- results[,c(1,5,6,7)]

# Transform
results_time$type = "time-series"
colnames(results_time) <- c('epochs', 'score', 'min', 'max', 'type')
results_features$type = "features"
colnames(results_features) <- c('epochs', 'score', 'min', 'max', 'type')
ggplot() +
    geom_line(data=results_time, aes(x = epochs, y=score, color=type)) +
    geom_ribbon(data=results_time, aes(x = epochs, ymax = max, ymin = min), alpha = 0.6, fill = "skyblue") +
    #scale_fill_manual(values=c("skyblue")) +
    #scale_color_manual(values=c("skyblue")) +
    geom_line(data=results_features, aes(x = epochs, y=score, color=type)) +
    geom_ribbon(data=results_features, aes(x = epochs, ymax = max, ymin = min), alpha = 0.6, fill = "lightsalmon") +
    scale_y_continuous(limits=c(0,1), breaks=seq(0, 1, 0.1)) +
    geom_hline(yintercept=0.25, linetype="dashed") +
    theme(text = element_text(size = 20),
          panel.grid.major.x = element_blank(),
          axis.ticks.y = element_line(colour="#e7e7e7"),
          panel.grid.major.y = element_line(size=0.1, color="#ededed"),
          panel.background = element_blank(),
          legend.title=element_blank(),

          legend.key=element_blank(),
          legend.position = c(0.15, 0.9)) +
    labs(x="Epochs", y="Score")
ggsave("./scripts/analysis/plots/validation_unfrozen_score.pdf")

###
# Training/Validation loss
###

# Read results
results <- read.csv(file = "./scripts/analysis/export_loss.csv", stringsAsFactors = FALSE)
results_features <- results[,c(1,2,3,4)]
results_time <- results[,c(1,5,6,7)]

# Transform
results_time$type = "time-series"
colnames(results_time) <- c('epochs', 'score', 'min', 'max', 'type')
results_features$type = "features"
colnames(results_features) <- c('epochs', 'score', 'min', 'max', 'type')
ggplot() +
    #scale_fill_manual(values=c("skyblue")) +
    #scale_color_manual(values=c("skyblue")) +
    geom_line(data=results_features, aes(x = epochs, y=score, color=type)) +
    geom_ribbon(data=results_features, aes(x = epochs, ymax = max, ymin = min), alpha = 0.6, fill = "lightsalmon") +
    geom_line(data=results_time, aes(x = epochs, y=score, color=type)) +
    geom_ribbon(data=results_time, aes(x = epochs, ymax = max, ymin = min), alpha = 0.6, fill = "skyblue") +
    scale_y_continuous(limits=c(0,16), breaks=seq(0, 16, 1)) +
    theme(text = element_text(size = 20),
          panel.grid.major.x = element_blank(),
          axis.ticks.y = element_line(colour="#e7e7e7"),
          panel.grid.major.y = element_line(size=0.1, color="#ededed"),
          panel.background = element_blank(),
          legend.title=element_blank(),
          legend.key=element_blank(),
          legend.position = c(0.15, 0.9)) +
    labs(x="Epochs", y="Loss")
ggsave("./scripts/analysis/plots/validation_unfrozen_loss.pdf")

###
# Test score
###
results <- read.csv(file = "./scripts/analysis/export_test.csv", stringsAsFactors = FALSE)[,c("data","Test.Accuracy")]
ggplot(results, aes(y = Test.Accuracy, fill=data)) +
    geom_boxplot(width=0.4) +
    scale_fill_manual(values=c("lightsalmon", "skyblue")) +
    xlim(-1,1) +
    facet_wrap(~fct_relevel(data, "time-series", "features"), scales = "free_x", strip.position = "bottom") +
    scale_y_continuous(limits=c(0,1), breaks=seq(0, 1, 0.1)) +
    geom_hline(yintercept=0.25, linetype="dashed") +
    theme(text = element_text(size = 20),
          panel.spacing = unit(0, "lines"),
          axis.ticks.y = element_line(colour="#e7e7e7"),
          panel.grid.major.y = element_line(size=0.1, color="#ededed"),
          strip.background = element_blank(),
          panel.background = element_blank(),
          strip.placement = "outside",
          axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          legend.position="none")
ggsave("./scripts/analysis/plots/test_optimisation_score.pdf")

###
# Training/validation score with gradients
###

# Read results
results <- read.csv(file = "./scripts/analysis/export_score.csv", stringsAsFactors = FALSE)[,c(1, seq(2,33,by=3))]

# Transform
transformed <- data.frame(matrix(ncol = 4, nrow = 0))
colnames(transformed) <- c('epochs', 'score', 'frozen', 'varying_frozen')

extract_frozen_lower <- function(name) {
    split <- unlist(strsplit(name, "[.]"))
    return(strtoi(tail(split,n=2)[1]))
}

extract_frozen_upper <- function(name) {
    split <- unlist(strsplit(name, "[.]"))
    return(strtoi(gsub("\\D", "", tail(split,n=1))))
}

for(i in 1:nrow(results)) {
    for(j in 2:ncol(results)) {
        frozen_lower <- extract_frozen_lower(colnames(results)[j])
        frozen_upper <- extract_frozen_upper(colnames(results)[j])
        transformed[nrow(transformed)+1,] <- c(results[i, 1], results[i, j], paste(frozen_lower, frozen_upper, sep="-"), frozen_lower)
    }
}
transformed$score <- as.double(transformed$score)
transformed$epochs <- as.numeric(transformed$epochs)
transformed$varying_frozen <- as.numeric(transformed$varying_frozen)

# Legend
labels = sapply(sort(unique(transformed$varying_frozen)), function(vary) {
    return(paste(vary,"12", sep="-"))
    #return(paste("1", vary, sep="-"))
})
values = c(1,11)

# Plot 
ggplot(data=transformed, aes(x=epochs, y=score, color=varying_frozen, group=frozen)) +
    geom_line() +
    geom_hline(yintercept=0.25, linetype="dashed") +
    #scale_color_gradientn(limits = c(0,12), breaks=values,labels=labels[values], colours=colorRampPalette(c("skyblue", "blue4"))(12)) +
    scale_color_gradientn(limits = c(0,12), breaks=values,labels=labels[values], colours=colorRampPalette(c("lightsalmon", "red4"))(12)) +
    scale_y_continuous(limits=c(0,1), breaks=seq(0, 1, 0.1)) +
    theme(text = element_text(size = 20),
          panel.grid.major.x = element_blank(),
          axis.ticks.y = element_line(colour="#e7e7e7"),
          panel.grid.major.y = element_line(size=0.1, color="#ededed"),
          panel.background = element_blank(),
          legend.background=element_rect(color="transparent"),
          legend.title = element_text(size=12),
          legend.text = element_text(size=12),
          legend.position = c(0.12, 0.87)) +
    labs(x="Epochs", y="Score", color="Frozen layers")
ggsave("./scripts/analysis/plots/validation_unfreezing_shallow-deep_features_score.pdf")

###
# Training/validation loss with gradients
###

# Read results
results <- read.csv(file = "./scripts/analysis/export_loss.csv", stringsAsFactors = FALSE)[,c(1, seq(2,33,by=3))]

# Transform
transformed <- data.frame(matrix(ncol = 4, nrow = 0))
colnames(transformed) <- c('epochs', 'score', 'frozen', 'varying_frozen')

extract_frozen_lower <- function(name) {
    split <- unlist(strsplit(name, "[.]"))
    return(strtoi(tail(split,n=2)[1]))
}

extract_frozen_upper <- function(name) {
    split <- unlist(strsplit(name, "[.]"))
    return(strtoi(gsub("\\D", "", tail(split,n=1))))
}

for(i in 1:nrow(results)) {
    for(j in 2:ncol(results)) {
        frozen_lower <- extract_frozen_lower(colnames(results)[j])
        frozen_upper <- extract_frozen_upper(colnames(results)[j])
        transformed[nrow(transformed)+1,] <- c(results[i, 1], results[i, j], paste(frozen_lower, "-", frozen_upper, sep=""), frozen_lower)
    }
}
transformed$score <- as.double(transformed$score)
transformed$epochs <- as.numeric(transformed$epochs)
transformed$varying_frozen <- as.numeric(transformed$varying_frozen)

# Plot 
ggplot(data=transformed, aes(x=epochs, y=score, color=varying_frozen, group=frozen)) +
    geom_line() +
    scale_y_continuous(limits=c(0,10), breaks=seq(0, 10, 1)) +
    #scale_color_gradientn(limits = c(0, 12), breaks=values,labels=labels[values], colours=colorRampPalette(c("skyblue", "blue4"))(12)) +
    scale_color_gradientn(limits = c(0, 12), breaks=values,labels=labels[values], colours=colorRampPalette(c("lightsalmon", "red4"))(12)) +
    theme(text = element_text(size = 20),
          panel.grid.major.x = element_blank(),
          axis.ticks.y = element_line(colour="#e7e7e7"),
          panel.grid.major.y = element_line(size=0.1, color="#ededed"),
          panel.background = element_blank(),
          legend.key=element_blank(),
          legend.title = element_text(size=12),
          legend.text = element_text(size=12),
          legend.position = c(0.15, 0.87)) +
    labs(x="Epochs", y="Loss", color="Frozen layers")
ggsave("./scripts/analysis/plots/validation_unfreezing_shallow-deep_features_loss.pdf")

###
# Test score with gradients
###

# Read results
results <- read.csv(file = "./scripts/analysis/export_test.csv", stringsAsFactors = FALSE)
results <- results[,c("freeze_lower", "freeze_upper", "Test.Accuracy")]
results$group <- ""
results$freeze_varying <- results$freeze_upper

# Transform
for(i in 1:nrow(results)) {
    results[i, "group"] <- paste(results[i, "freeze_lower"], "-", results[i, "freeze_upper"], sep="")
}

# Plot 
ggplot(results, aes(x=reorder(group, -freeze_varying), y=Test.Accuracy, group=group)) +
    #geom_boxplot(width=0.4, color="black", fill="skyblue") +
    geom_boxplot(width=0.4, color="black", fill="lightsalmon") +
    scale_y_continuous(limits=c(0,1), breaks=seq(0, 1, 0.1)) +
    geom_hline(yintercept=0.25, linetype="dashed") +
    theme(text = element_text(size = 20),
          panel.spacing = unit(0, "lines"),
          axis.ticks.y = element_line(colour="#e7e7e7"),
          axis.text.x = element_text(size=12),
          panel.grid.major.y = element_line(size=0.1, color="#ededed"),
          strip.background = element_blank(),
          panel.background = element_blank(),
          strip.placement = "outside",
          legend.position = "none") +
    labs(x="Frozen layers", y="Score")
ggsave("./scripts/analysis/plots/test_unfreezing_deep-shallow_features.pdf")
