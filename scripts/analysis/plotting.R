library(ggplot2)
library(forcats)

###
# Training/validation score
###

# Read results
results <- read.csv(file = "./scripts/analysis/export_score.csv", stringsAsFactors = FALSE)
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
    scale_fill_manual(values=c("skyblue")) +
    scale_color_manual(values=c("skyblue")) +
    #geom_line(data=results_features, aes(x = epochs, y=score, color=type)) +
    #geom_ribbon(data=results_features, aes(x = epochs, ymax = max, ymin = min), alpha = 0.6, fill = "lightsalmon") +
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
ggsave("./scripts/analysis/plots/validation_overfitting_score.pdf")

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
    geom_line(data=results_time, aes(x = epochs, y=score, color=type)) +
    geom_ribbon(data=results_time, aes(x = epochs, ymax = max, ymin = min), alpha = 0.6, fill = "skyblue") +
    #scale_fill_manual(values=c("skyblue")) +
    #scale_color_manual(values=c("skyblue")) +
    geom_line(data=results_features, aes(x = epochs, y=score, color=type)) +
    geom_ribbon(data=results_features, aes(x = epochs, ymax = max, ymin = min), alpha = 0.6, fill = "lightsalmon") +
    scale_y_continuous(limits=c(0,10), breaks=seq(0, 10, 1)) +
    theme(text = element_text(size = 20),
          panel.grid.major.x = element_blank(),
          axis.ticks.y = element_line(colour="#e7e7e7"),
          panel.grid.major.y = element_line(size=0.1, color="#ededed"),
          panel.background = element_blank(),
          legend.title=element_blank(),
          legend.key=element_blank(),
          legend.position = c(0.15, 0.9)) +
    labs(x="Epochs", y="Loss")
ggsave("./scripts/analysis/plots/training_optimisation_loss.pdf")

###
# Test score
###
results <- read.csv(file = "./scripts/analysis/export_test.csv", stringsAsFactors = FALSE)[,c(6,9)]
colnames(results) <- c('type', 'score')
ggplot(results, aes(y = score, fill=type, color = type)) +
    geom_boxplot(width=0.4) +
    xlim(-1,1) +
    facet_wrap(~fct_relevel(type, "time-series", "features"), scales = "free_x", strip.position = "bottom") +
    scale_y_continuous(limits=c(0,1), breaks=seq(0, 1, 0.1)) +
    geom_hline(yintercept=0.25, linetype="dashed") +
    scale_fill_manual(values=c("lightsalmon", "skyblue")) +
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
ggsave("./scripts/analysis/plots/test_overfitting_score.pdf")
