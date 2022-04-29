library(ggplot2)
library(ggthemes)
library(forcats)

# Training/Validation score
results_time <- read.csv(file = "./scripts/analysis/export_features.csv", stringsAsFactors = FALSE)
results_time$type = "time-series"
colnames(results_time) <- c('epochs', 'score', 'min', 'max', 'type')
results_features <- read.csv(file = "./scripts/analysis/export_time.csv", stringsAsFactors = FALSE)
results_features$type = "features"
colnames(results_features) <- c('epochs', 'score', 'min', 'max', 'type')
ggplot() +
    geom_line(data=results_time, aes(x = epochs, y=score, color="skyblue")) +
    geom_ribbon(data=results_time, aes(x = epochs, ymax = max, ymin = min), alpha = 0.6, fill = "skyblue") +
    geom_line(data=results_features, aes(x = epochs, y=score, color="lightsalmon")) +
    geom_ribbon(data=results_features, aes(x = epochs, ymax = max, ymin = min), alpha = 0.6, fill = "lightsalmon") +
    scale_y_continuous(limits=c(0,1), breaks=seq(0, 1, 0.1)) +
    scale_color_manual("",values = c("skyblue", "lightsalmon"), labels = c("time-series", "features"))+
    theme(text = element_text(size = 30),
          panel.grid.major.x = element_blank(),
          axis.ticks.y = element_line(colour="#e7e7e7"),
          panel.grid.major.y = element_line(size=0.1, color="#ededed"),
          panel.background = element_blank(),
          legend.title=element_blank(),
          legend.key=element_blank(),
          legend.position = c(0.86, 0.94)) +
    labs(x="Epochs", y="Score")
ggsave("./scripts/analysis/plots/training_optimisation.pdf")

# Test score
results <- read.csv(file = "./scripts/analysis/test_export.csv", stringsAsFactors = FALSE)
colnames(results) <- c('type', 'empty', 'score')
results$type <- unlist(lapply(results$type, function(group) {
    if(grepl("signal", group, fixed = TRUE)) {
        return("time-series")
    } else {
        return("features")
    }
}))
results$type[2] <- "time-series"
results$type[3] <- "time-series"
results$type[4] <- "time-series"
ggplot(results, aes(y = score, fill=type, color = type)) +
    geom_boxplot() +
    facet_wrap(~fct_relevel(type, "time-series", "features"), scales = "free_x", strip.position = "bottom") +
    scale_y_continuous(limits=c(0,1), breaks=seq(0, 1, 0.1)) +
    scale_fill_manual(values=c("lightsalmon", "skyblue")) +
    theme(panel.spacing = unit(0, "lines"),
          strip.background = element_blank(),
          panel.background = element_blank(),
          strip.placement = "outside",
          axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          legend.position="none")
ggsave("./scripts/analysis/plots/test_optimisation.pdf")
