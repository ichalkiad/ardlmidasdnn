#################################################################################

# Ioannis Chalkiadakis
# August, 2021

# Resilient ML
# Script to perform 2-Stage Least Squares Regressions for Instrumental Variables
# regression

#################################################################################

library("lmtest")
library("stats")
library("logr")
library("car")
library("tseries")
library("KSgeneral")
library("pracma")
library("ggplot2")
library("forecast")
library("vsgoftest")

# If regression model for IV is bad, then we do not proceed to regress with fitted IV

fit_reduced_lm <- function(m) {

  # get p-values
  pvals <- summary(m)$coefficients[,4]
  # keep statistically significant coefficients
  coefs <- m$coefficients
  coefs[pvals > 0.05] <- 0
  # get fitted values using signif. coefficients
  m$coefficients <- coefs
  fitted_vals <- predict(m)

  return(fitted_vals)
}

# daily_sumary <- "median"
entropy_type <- "entropy_abs_strength_BTC"
asset <- "BTC"

DIR <- "/tmp/"
dir.create(sprintf("%s%s", DIR, "casestudy1"))

smoothing_win <- 180 # 7, 21, 90, 180
DIR_out <- sprintf("%scasestudy1/koyck_results_smooth%s/", DIR, smoothing_win)
print(DIR_out)
dir.create(DIR_out)


stage1_arima_max <- 5
stepsize <- 30 # 7
window_size <- 210 # 31

load(file<-sprintf("%svolume_joint_weighted_per_asset_cubic_interpolated_median_smooth%sdays_timeseries.rdata", DIR, smoothing_win))
for (asset in list("ETH", "BTC")) { # , "BTC"
     for (entropy_type in list(sprintf("entropy_abs_strength_%s", asset),
                               sprintf("entropy_negative_%s", asset),
                               sprintf("entropy_positive_%s", asset) )) {


          DIR_out_st1 <- sprintf("%sstage1_%s/", DIR_out, entropy_type)
          print(DIR_out_st1)
          dir.create(DIR_out_st1)

          logging <- sprintf("%s%s_koyck_log_smooth%s_vstests", DIR_out_st1, entropy_type, smoothing_win)
          logf <- log_open(logging)


          response <- rpy2df[, entropy_type]
          bert <- rpy2df[, sprintf("BERT_%s", asset)]
          vader <- rpy2df[, sprintf("VADER_%s", asset)]

          # keep same range of data for all time-series
          response <- scale(response[smoothing_win:length(response)])
          bert <- scale(bert[smoothing_win:length(bert)])
          vader <- scale(vader[smoothing_win:length(vader)])

          if (length(response) < 200) {
              print("TOO SHORT TIMESERIΕS???")
              stepsize <- 7
              window_size <- 31
          } else {
              stepsize <- 30
              window_size <- 210
          }

          best_IV_regr <- rep("-1000", 1 + floor((length(response)-window_size)/stepsize))
          class(best_IV_regr) <- "character"
          days_win <- rep("-1000", 1 + floor((length(response)-window_size)/stepsize))
          class(days_win) <- "character"

          coef <- rep(-1000, 1 + floor((length(response)-window_size)/stepsize))
          intercepts <- rep(-1000, 1 + floor((length(response)-window_size)/stepsize))
          # BERT
          betas1 <- rep(-1000, 1 + floor((length(response)-window_size)/stepsize))
          # VADER
          betas2 <- rep(-1000, 1 + floor((length(response)-window_size)/stepsize))
          phis <- rep(-1000, 1 + floor((length(response)-window_size)/stepsize))

          rseA <- rep(-1000, 1 + floor((length(response)-window_size)/stepsize))
          rsePhi <- rep(-1000, 1 + floor((length(response)-window_size)/stepsize))
          rseB1 <- rep(-1000, 1 + floor((length(response)-window_size)/stepsize))
          rseB2 <- rep(-1000, 1 + floor((length(response)-window_size)/stepsize))

          pvalA <- rep(-1000, 1 + floor((length(response)-window_size)/stepsize))
          pvalPhi <- rep(-1000, 1 + floor((length(response)-window_size)/stepsize))
          pvalB1 <- rep(-1000, 1 + floor((length(response)-window_size)/stepsize))
          pvalB2 <- rep(-1000, 1 + floor((length(response)-window_size)/stepsize))

          stage2_significance_BERT <- rep("NA", 1 + floor((length(response)-window_size)/stepsize))
          stage2_significance_VADER <- rep("NA", 1 + floor((length(response)-window_size)/stepsize))
          stage1_problematic_windows <- rep("NA", 1 + floor((length(response)-window_size)/stepsize))

          k <- 1
          faulty_win <- 0
          for (i in seq(1, length(response), stepsize) ) {
            reject_win <- FALSE
            if (as.Date(rpy2df$Dates[i]) < as.Date("2018-01-01")){
               print(rpy2df$Dates[i])
               next
            }
            log_print(sprintf("******************************  Window start: %s  *********************************************",
                              rpy2df$Dates[i]))
            print(i)
            stage2_m1 <- NA
            lmtest_m1 <- NA
            pm1 <- NA
            stage2_m2 <- NA
            lmtest_m2 <- NA
            pm2 <- NA
            stage2_m3 <- NA
            lmtest_m3 <- NA
            pm3 <- NA
            stage2_m4 <- NA
            lmtest_m4 <- NA
            pm4 <- NA
            stage2_m5 <- NA
            lmtest_m5 <- NA
            pm5 <- NA
            stage2_m6 <- NA
            lmtest_m6 <- NA
            pm6 <- NA
            best_m <- NA
            best_coef <- NA
            best_coef_pval <- NA
            best_coef_ci <- NA
            save_for_aic <- rep(NA, 6)

            ## STAGE I: regress Y on Y.t_1 and get residuals e_st1
            y <- response[i:(i+window_size-1)]
            # For Stage II
            x_B <- bert[i:(i+window_size-1)]
            x_V <- vader[i:(i+window_size-1)]

            if (length(y[!is.na(y)]) < 2) {
              print("Invalid values in window???")
              break
            }
            if (length(y[is.na(y)]) > 0) {
              print("Removing NAs")
              x_B <- x_B[!is.na(y)]
              x_V <- x_V[!is.na(y)]
              y <- y[!is.na(y)]
            }

            stage1_aic_diff <- rep("NA", stage1_arima_max)
            stage1_bestmodel <- rep("NA", 2)
            for (optimmethod in c("BFGS", "Nelder-Mead", "CG")) {
              # print(optimmethod)
              stage1_aic <- rep(Inf, stage1_arima_max)
              stage1_bestmodel[1] <- optimmethod
              for (iii in 1:stage1_arima_max) {
                  # print(iii)
                  pvals1 <- tryCatch({
                    stage1 <- arima(y, c(iii, 0, 0), optim.method = optimmethod, method="ML")
                    pvals1 <- 2*(1 - pnorm(abs(stage1$coef)/sqrt(diag(vcov(stage1)))))
                  }, error=function(pvals1){return (NA)}
                  )
                  if (sum(is.na(pvals1)) > 0 || stage1$code > 0){
                      print(pvals1)
                      print(stage1$code)
                      next
                  }
                  else{
                    stage1_aic[iii] <- AIC(stage1)
                  }
              }
              if (sum(is.na(stage1_aic)) == 0){
                 print("SAVE")
                 stage1_aic_idx <- order(stage1_aic)
                 stage1_bestmodel[2] <- stage1_aic_idx[1]
                 stage1_aic_diff <- diff(stage1_aic[stage1_aic_idx])
                 stage1_allmodels <- seq(1, stage1_arima_max, 1)[stage1_aic_idx]
                 log_print(stage1_allmodels)
                 break
              }
            }

            if (stage1_bestmodel[2]=="NA"){
              log_print(sprintf("Skipping window starting on %s", rpy2df$Dates[i]))
              print(sprintf("Skipping window starting on %s", rpy2df$Dates[i]))
              next
            }
            else{
              stage1 <- arima(y, c(strtoi(stage1_bestmodel[2]), 0, 0), optim.method = stage1_bestmodel[1], method="ML")
              pvals1 <- 2*(1 - pnorm(abs(stage1$coef)/sqrt(diag(vcov(stage1)))))
              stage1_significance_ar_params <- rep("NA", strtoi(stage1_bestmodel[2])+1)
              print("SAVE")
            }

            e_st1 <- residuals(stage1)

            # residuals Q-Q plots
            pdf(sprintf("%sstage1_residualsQQ_norm_%s_median_smooth%s_window%s_%s.pdf", DIR_out_st1, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]))
            qqPlot(e_st1, distribution="norm")
            dev.off()
            pdf(sprintf("%sstage1_residualsQQ_t_%s_median_smooth%s_window%s_%s.pdf", DIR_out_st1, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]))
            qqPlot(e_st1, distribution="t", df=3)
            dev.off()
            jpeg(sprintf("%sstage1_residualsQQ_norm_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_st1, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]))
            qqPlot(e_st1, distribution="norm")
            dev.off()
            jpeg(sprintf("%sstage1_residualsQQ_t_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_st1, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]))
            qqPlot(e_st1, distribution="t", df=3)
            dev.off()
            pdf(sprintf("%sstage1_residualsQQ_t15_%s_median_smooth%s_window%s_%s.pdf", DIR_out_st1, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]))
            qqPlot(e_st1, distribution="t", df=15)
            dev.off()
            pdf(sprintf("%sstage1_residualsQQ_t10_%s_median_smooth%s_window%s_%s.pdf", DIR_out_st1, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]))
            qqPlot(e_st1, distribution="t", df=10)
            dev.off()
            jpeg(sprintf("%sstage1_residualsQQ_t15_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_st1, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]))
            qqPlot(e_st1, distribution="t", df=15)
            dev.off()
            jpeg(sprintf("%sstage1_residualsQQ_t10_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_st1, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]))
            qqPlot(e_st1, distribution="t", df=10)
            dev.off()
            violindata <- data.frame(x=rep(sprintf("Window starting on %s", rpy2df$Dates[i]), length(e_st1)), y=c(e_st1))
            ggplot(violindata, aes(x=x, y=y)) + geom_violin(trim=FALSE) + geom_boxplot(width=0.2) +
              labs(title=sprintf("Residuals - window starting on %s", rpy2df$Dates[i])) + theme_classic()
            ggsave(sprintf("%sstage1_residualsViolin_%s_median_smooth%s_window%s_%s.pdf", DIR_out_st1, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]), width = 7, height = 7, units = "in")
            ggplot(violindata, aes(x=x, y=y)) + geom_violin(trim=FALSE) + geom_boxplot(width=0.2) +
              labs(title=sprintf("Residuals - window starting on %s", rpy2df$Dates[i])) + theme_classic()
            ggsave(sprintf("%sstage1_residualsViolin_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_st1, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]), width = 7, height = 7, units = "in")

            pdf(sprintf("%sstage1_residualsVsfitted_%s_median_smooth%s_window%s_%s.pdf", DIR_out_st1, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]))
            fitval <- y - e_st1
            plot(e_st1, fitval, main="Residuals vs Fitted values", xlab="Residuals", ylab="Fitted values")
            dev.off()
            jpeg(sprintf("%sstage1_residualsVsfitted_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_st1, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]))
            plot(e_st1, fitval, main="Residuals vs Fitted values", xlab="Residuals", ylab="Fitted values", pch=19)
            dev.off()

            # residuals KPSS test for trend stationarity
            diff_ts <- FALSE
            kpss_result <- kpss.test(e_st1, null=c("Trend"), lshort = FALSE)
            log_print("STAGE 1 errors - KPSS tests for trend stationarity")
            if (kpss_result$p.value <= 0.001) {
               stage1_significance_kpss <- "***"
               diff_ts <- TRUE
            }
            else if (kpss_result$p.value <= 0.01) {
               stage1_significance_kpss <- "**"
               diff_ts <- TRUE
            }
            else if (kpss_result$p.value <= 0.05) {
               stage1_significance_kpss <- "*"
               diff_ts <- TRUE
            }
            else if (kpss_result$p.value <= 0.1) {
               stage1_significance_kpss <- "."
            }
            else {  # if (kpss_result$p.value > 0.1)
               stage1_significance_kpss <- ".."
            }
            log_print(kpss_result)
            log_print(stage1_significance_kpss)
            diff_times <- 0
            while (diff_ts){
              diff_ts <- FALSE
              diff_times <- diff_times + 1
              if (diff_times > 4){
                reject_win <- TRUE
                break
              }

              log_print("KPSS failed - difference time-series and refit best model")
              yy <- diff(y)
              stage1 <- arima(yy, c(strtoi(stage1_bestmodel[2]), 0, 0), optim.method = stage1_bestmodel[1], method="ML")
              pvals1 <- 2*(1 - pnorm(abs(stage1$coef)/sqrt(diag(vcov(stage1)))))
              e_st1 <- residuals(stage1)
              # residuals Q-Q plots
              pdf(sprintf("%sstage1_residualsQQ_norm_%s_median_smooth%s_window%s_%s_post_diff.pdf", DIR_out_st1, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st1, distribution="norm")
              dev.off()
              pdf(sprintf("%sstage1_residualsQQ_t_%s_median_smooth%s_window%s_%s_post_diff.pdf", DIR_out_st1, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st1, distribution="t", df=3)
              dev.off()
              jpeg(sprintf("%sstage1_residualsQQ_norm_%s_median_smooth%s_window%s_%s_post_diff.jpeg", DIR_out_st1, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st1, distribution="norm")
              dev.off()
              jpeg(sprintf("%sstage1_residualsQQ_t_%s_median_smooth%s_window%s_%s_post_diff.jpeg", DIR_out_st1, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st1, distribution="t", df=3)
              dev.off()
              pdf(sprintf("%sstage1_residualsQQ_t15_%s_median_smooth%s_window%s_%s_post_diff.pdf", DIR_out_st1, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st1, distribution="t", df=15)
              dev.off()
              pdf(sprintf("%sstage1_residualsQQ_t10_%s_median_smooth%s_window%s_%s_post_diff.pdf", DIR_out_st1, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st1, distribution="t", df=10)
              dev.off()
              jpeg(sprintf("%sstage1_residualsQQ_t15_%s_median_smooth%s_window%s_%s_post_diff.jpeg", DIR_out_st1, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st1, distribution="t", df=15)
              dev.off()
              jpeg(sprintf("%sstage1_residualsQQ_t10_%s_median_smooth%s_window%s_%s_post_diff.jpeg", DIR_out_st1, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st1, distribution="t", df=10)
              dev.off()
              violindata <- data.frame(x=rep(sprintf("Window starting on %s", rpy2df$Dates[i]), length(e_st1)), y=c(e_st1))
              ggplot(violindata, aes(x=x, y=y)) + geom_violin(trim=FALSE) + geom_boxplot(width=0.2) +
                labs(title=sprintf("Residuals - window starting on %s", rpy2df$Dates[i])) + theme_classic()
              ggsave(sprintf("%sstage1_residualsViolin_%s_median_smooth%s_window%s_%s_post_diff.pdf", DIR_out_st1, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]), width = 7, height = 7, units = "in")
              ggplot(violindata, aes(x=x, y=y)) + geom_violin(trim=FALSE) + geom_boxplot(width=0.2) +
                labs(title=sprintf("Residuals - window starting on %s", rpy2df$Dates[i])) + theme_classic()
              ggsave(sprintf("%sstage1_residualsViolin_%s_median_smooth%s_window%s_%s_post_diff.jpeg", DIR_out_st1, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]), width = 7, height = 7, units = "in")

              pdf(sprintf("%sstage1_residualsVsfitted_%s_median_smooth%s_window%s_%s_post_diff.pdf", DIR_out_st1, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              fitval <- yy - e_st1
              plot(e_st1, fitval, main="Residuals vs Fitted values", xlab="Residuals", ylab="Fitted values", pch=19)
              dev.off()
              jpeg(sprintf("%sstage1_residualsVsfitted_%s_median_smooth%s_window%s_%s_post_diff.jpeg", DIR_out_st1, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              plot(e_st1, fitval, main="Residuals vs Fitted values", xlab="Residuals", ylab="Fitted values", pch=19)
              dev.off()


              # residuals KPSS test for trend stationarity
              kpss_result <- kpss.test(e_st1, null=c("Trend"), lshort = FALSE)
              log_print("STAGE 1 errors - KPSS tests for trend stationarity")
              if (kpss_result$p.value <= 0.001) {
                 stage1_significance_kpss <- "***"
                 diff_ts <- TRUE
              }
              else if (kpss_result$p.value <= 0.01) {
                 stage1_significance_kpss <- "**"
                 diff_ts <- TRUE
              }
              else if (kpss_result$p.value <= 0.05) {
                 stage1_significance_kpss <- "*"
                 diff_ts <- TRUE
              }
              else if (kpss_result$p.value <= 0.1) {
                 stage1_significance_kpss <- "."
              }
              else {  # if (kpss_result$p.value > 0.1)
                 stage1_significance_kpss <- ".."
              }
              log_print(kpss_result)
              log_print(stage1_significance_kpss)

            }

            if (reject_win) {
              faulty_win <- faulty_win + 1
              stage1_problematic_windows[faulty_win] <- rpy2df$Dates[i]
              print(rpy2df$Dates[i])
              next
            }

            # KS test - KSgeneral needs detrended time-series
            e_st1_detrend <- detrend(c(e_st1))
            ks_result <- KSgeneral::cont_ks_test(e_st1_detrend, "pnorm")
            log_print("STAGE 1 errors - KS tests for normality")
            if (ks_result$p <= 0.001) {
               stage1_significance_ks <- "***"
            }
            else if (ks_result$p <= 0.01) {
               stage1_significance_ks <- "**"
            }
            else if (ks_result$p <= 0.05) {
               stage1_significance_ks <- "*"
            }
            else if (ks_result$p <= 0.1) {
               stage1_significance_ks <- "."
            }
            else if (ks_result$p > 0.1) {
               stage1_significance_ks <- ".."
            }
            log_print(ks_result)
            log_print(stage1_significance_ks)

            # VS test
            vs_result <- vs.test(x=c(e_st1), densfun="dnorm", simulate.p.value=TRUE)
            log_print("STAGE 1 errors - VS tests for normality")
            if (vs_result$p.value <= 0.001) {
               stage1_significance_vs <- "***"
            }
            else if (vs_result$p.value <= 0.01) {
               stage1_significance_vs <- "**"
            }
            else if (vs_result$p.value <= 0.05) {
               stage1_significance_vs <- "*"
            }
            else if (vs_result$p.value <= 0.1) {
               stage1_significance_vs <- "."
            }
            else if (vs_result$p.value > 0.1) {
               stage1_significance_vs <- ".."
            }
            log_print(vs_result)
            log_print(stage1_significance_vs)

            # Density plots
            pdf(sprintf("%sstage1_DensityPlots_%s_median_smooth%s_window%s_%s.pdf", DIR_out_st1, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]))
            plot(density(e_st1_detrend), col=adjustcolor("red",alpha.f=0.8), main="")
            if ((sd(e_st1_detrend)-0.1) < (sd(e_st1_detrend)-0.01)){
                x_standardnormal <- rnorm(500, mean = 0, sd=0.01)
            }
            else{
                x_standardnormal <- rnorm(500, mean = 0, sd=0.01)
            }
            polygon(density(x_standardnormal), col=adjustcolor("green", alpha.f=0.1))
            legend("topright", legend = c("Residual density", "Normal density"), col = c("red", "green"), lty = 1)
            dev.off()
            jpeg(sprintf("%sstage1_DensityPlots_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_st1, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]))
            plot(density(e_st1_detrend), col=adjustcolor("red",alpha.f=0.8), main="")
            polygon(density(x_standardnormal), col=adjustcolor("green", alpha.f=0.1))
            legend("topright", legend = c("Residual density", "Normal density"), col = c("red", "green"), lty = 1)
            dev.off()


            # ACF/PACF
            pdf(sprintf("%sstage1_ACF_%s_median_smooth%s_window%s_%s.pdf", DIR_out_st1, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]))
            acf(e_st1, main = sprintf("Stage I errors - window start: %s", rpy2df$Dates[i]))
            dev.off()
            pdf(sprintf("%sstage1_PACF_%s_median_smooth%s_window%s_%s.pdf", DIR_out_st1, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]))
            pacf(e_st1, main = sprintf("Stage I errors - window start: %s", rpy2df$Dates[i]))
            dev.off()
            jpeg(sprintf("%sstage1_ACF_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_st1, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]))
            acf(e_st1, main = sprintf("Stage I errors - window start: %s", rpy2df$Dates[i]))
            dev.off()
            jpeg(sprintf("%sstage1_PACF_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_st1, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]))
            pacf(e_st1, main = sprintf("Stage I errors - window start: %s", rpy2df$Dates[i]))
            dev.off()

            coefs <- stage1$coef
            coefs[pvals1 > 0.05] <- 0
            # coefs[1]: slope
            if (coefs[1] > 0) {
              log_print("STAGE 1")
              log_print(summary(stage1))
            }
            else{
              faulty_win <- faulty_win + 1
              stage1_problematic_windows[faulty_win] <- rpy2df$Dates[i]
              next
            }
            for (jjj in 1:(strtoi(stage1_bestmodel[2])+1)){
              # last is intercept
              if (pvals1[jjj] <= 0.001) {
                 stage1_significance_ar_params[jjj] <- "***"
              }
              else if (pvals1[jjj] <= 0.01) {
                 stage1_significance_ar_params[jjj] <- "**"
              }
              else if (pvals1[jjj] <= 0.05) {
                 stage1_significance_ar_params[jjj] <- "*"
              }
              else if (pvals1[jjj] <= 0.1) {
                 stage1_significance_ar_params[jjj] <- "."
              }
              else { # if (pvals1[1] > 0.1)
                 stage1_significance_ar_params[jjj] <- ".."
              }
            }

            log_print("Stage I")
            log_print("**********")
            log_print(stage1)
            log_print("p-values")
            log_print(pvals1)
            log_print("**********")
            save_for_aic[1] <- AIC(stage1)


            ### SAVE STAGE I
            data_test_stage1 <- data.frame(window=rep(rpy2df$Dates[i], 1+strtoi(stage1_bestmodel[2])),
                                           stage1_significance_ar_params=stage1_significance_ar_params,
                                           stage1_significance_kpss=rep(stage1_significance_kpss, 1+strtoi(stage1_bestmodel[2])),
                                           stage1_significance_ks=rep(stage1_significance_ks, 1+strtoi(stage1_bestmodel[2])),
                                           stage1_significance_vs=rep(stage1_significance_vs, 1+strtoi(stage1_bestmodel[2])))
            write.csv(data_test_stage1, sprintf("%sstage1_test_significance_%s_median_smooth%s_window%s_%s_VS_SAMPLING.csv",
                                                DIR_out_st1, entropy_type, smoothing_win, window_size, rpy2df$Dates[i]),
                                                row.names = FALSE)
            stage1_outlist <- list(stage1_aic[stage1_aic_idx], stage1_bestmodel, stage1_aic_diff, stage1_allmodels)
            save(stage1_outlist, file=sprintf("%s%s_stage1_windowStart_%s_best_model.rdata",
                                     DIR_out_st1, entropy_type, rpy2df$Dates[i]))



            ## STAGE II

            if (diff_ts || diff_times > 0) {
              print(length(x_B))
              print(length(e_st1))
              x_B <- x_B[(diff_times+1):length(x_B)]
              x_V <- x_V[(diff_times+1):length(x_V)]
            }

            # M2: BERT, IV regr. BERT

            DIR_out_m2 <- sprintf("%scasestudy1/koyck_results_smooth%s/M2/", DIR, smoothing_win)
            print(DIR_out_m2)
            dir.create(DIR_out_m2)

            # regression for instrumental variable
            stage2a <- lm(e_st1 ~ x_B)
            log_print("########## Stage II - M2 IV fit ########## ")
            log_print("**********")
            log_print(summary(stage2a))
            log_print("**********")
            # get IV
            m2_IV <- fit_reduced_lm(stage2a)    # fitted(stage2a)

            if (abs(sum(m2_IV)) > 1e-16) {
              #print(m2_IV)

              stage2_m2 <- lm(e_st1[2:length(e_st1)] ~  m2_IV[1:length(m2_IV)-1] + x_B[2:length(x_B)])
              if (sum(is.na(stage2_m2$coefficients)) > 0){
                 next
              }
              e_st2m2 <- residuals(stage2_m2)

              # ACF/PACF
              pdf(sprintf("%sstage2_ACF_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m2, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              acf(e_st2m2, main = sprintf("Stage 2 errors - window start: %s", rpy2df$Dates[i]))
              dev.off()
              pdf(sprintf("%sstage2_PACF_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m2, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              pacf(e_st2m2, main = sprintf("Stage 2 errors - window start: %s", rpy2df$Dates[i]))
              dev.off()
              jpeg(sprintf("%sstage2_ACF_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m2, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              acf(e_st2m2, main = sprintf("Stage 2 errors - window start: %s", rpy2df$Dates[i]))
              dev.off()
              jpeg(sprintf("%sstage2_PACF_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m2, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              pacf(e_st2m2, main = sprintf("Stage 2 errors - window start: %s", rpy2df$Dates[i]))
              dev.off()

              # residuals Q-Q plots
              pdf(sprintf("%sstage2_residualsQQ_norm_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m2, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m2, distribution="norm")
              dev.off()
              pdf(sprintf("%sstage2_residualsQQ_t_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m2, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m2, distribution="t", df=3)
              dev.off()
              jpeg(sprintf("%sstage2_residualsQQ_norm_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m2, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m2, distribution="norm")
              dev.off()
              jpeg(sprintf("%sstage2_residualsQQ_t_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m2, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m2, distribution="t", df=3)
              dev.off()
              pdf(sprintf("%sstage2_residualsQQ_t15_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m2, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m2, distribution="t", df=15)
              dev.off()
              pdf(sprintf("%sstage2_residualsQQ_t10_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m2, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m2, distribution="t", df=10)
              dev.off()
              jpeg(sprintf("%sstage2_residualsQQ_t15_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m2, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m2, distribution="t", df=15)
              dev.off()
              jpeg(sprintf("%sstage2_residualsQQ_t10_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m2, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m2, distribution="t", df=10)
              dev.off()
              violindata <- data.frame(x=rep(sprintf("Window starting on %s", rpy2df$Dates[i]), length(e_st2m2)), y=c(e_st2m2))
              ggplot(violindata, aes(x=x, y=y)) + geom_violin(trim=FALSE) + geom_boxplot(width=0.2) +
                labs(title=sprintf("Residuals - window starting on %s", rpy2df$Dates[i])) + theme_classic()
              ggsave(sprintf("%sstage2_residualsViolin_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m2, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]), width = 7, height = 7, units = "in")
              violindata <- data.frame(x=rep(sprintf("Window starting on %s", rpy2df$Dates[i]), length(e_st2m2)), y=c(e_st2m2))
              ggplot(violindata, aes(x=x, y=y)) + geom_violin(trim=FALSE) + geom_boxplot(width=0.2) +
                labs(title=sprintf("Residuals - window starting on %s", rpy2df$Dates[i])) + theme_classic()
              ggsave(sprintf("%sstage2_residualsViolin_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m2, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]), width = 7, height = 7, units = "in")

              pdf(sprintf("%sstage2_residualsVsfitted_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m2, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              plot(e_st2m2, m2_IV[2:length(m2_IV)], main="Residuals vs Fitted Instrumental Variable", xlab="Residuals", ylab="IV", pch=19)
              dev.off()
              jpeg(sprintf("%sstage2_residualsVsfitted_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m2, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              plot(e_st2m2, m2_IV[2:length(m2_IV)], main="Residuals vs Fitted Instrumental Variable", xlab="Residuals", ylab="IV", pch=19)
              dev.off()


              log_print("########## Stage II - M2 ########## ")
              log_print("**********")
              log_print(summary(stage2_m2))
              log_print("**********")
              # LM test for autocorrelation
              lmtest_m2 <- bgtest(stage2_m2)
              log_print(lmtest_m2)
              pm2 <- lmtest_m2$p.val
              # the null hypothesis is that there is no serial correlation of any order up to p
              if (pm2 <= 0.05) {
                log_print("M2 has bad IV")
              } else {
                log_print("M2 has good IV")
                log_print(pm2)
                if (length(which(is.na(stage2_m2$coefficients))) > 0){
                  print("**************** CHECK MODEL 2 REGRESSION *****************")
                }
                # pvals <- summary(stage2_m2)$coefficients[,4]
                # coefs <- stage2_m2$coefficients
                # coefs[pvals > 0.05] <- 0
                coefs <- stage2_m2$coefficients
                pvals <- rep("NA", length(coefs))
                rem <- 0
                for (kkk in 1:length(coefs)){
                  if (is.na(coefs[kkk])){
                     pvals[kkk] <- "NA"
                     rem <- rem + 1
                  }
                  else {
                     pvals[kkk] <- summary(stage2_m2)$coefficients[(kkk-rem), 4]
                  }
                }
                coefs[which(is.na(coefs))] <- -1000
                coefs[pvals > 0.05] <- 0
                # BERT coef
                if (coefs[3] > 0) {
                  #print(summary(stage2_m2))
                  save_for_aic[2] <- AIC(stage2_m2)
                }
              }
            } else {
              log_print("All coefficients of IV regression were insignificant, change model? - skipping regression with fitted IV")
              log_print(abs(sum(m2_IV)))
            }

            # M3: VADER, IV regr. VADER
            DIR_out_m3 <- sprintf("%scasestudy1/koyck_results_smooth%s/M3/", DIR, smoothing_win)
            print(DIR_out_m3)
            dir.create(DIR_out_m3)

            # regression for instrumental variable
            stage2b <- lm(e_st1 ~ x_V)
            log_print("########## Stage II - M3 IV fit ########## ")
            log_print("**********")
            log_print(summary(stage2b))
            log_print("**********")
            # get IV
            m3_IV <- fit_reduced_lm(stage2b)  # fitted(stage2b)

            if (abs(sum(m3_IV)) > 1e-16) {
              # print(m3_IV)

              stage2_m3 <- lm(e_st1[2:length(e_st1)] ~  m3_IV[1:length(m3_IV)-1] + x_V[2:length(x_V)])
              if (sum(is.na(stage2_m3$coefficients)) > 0){
                 next
              }
              e_st2m3 <- residuals(stage2_m3)

              # ACF/PACF
              pdf(sprintf("%sstage2_ACF_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m3, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              acf(e_st2m3, main = sprintf("Stage 2 errors - window start: %s", rpy2df$Dates[i]))
              dev.off()
              pdf(sprintf("%sstage2_PACF_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m3, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              pacf(e_st2m3, main = sprintf("Stage 2 errors - window start: %s", rpy2df$Dates[i]))
              dev.off()
              jpeg(sprintf("%sstage2_ACF_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m3, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              acf(e_st2m3, main = sprintf("Stage 2 errors - window start: %s", rpy2df$Dates[i]))
              dev.off()
              jpeg(sprintf("%sstage2_PACF_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m3, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              pacf(e_st2m3, main = sprintf("Stage 2 errors - window start: %s", rpy2df$Dates[i]))
              dev.off()

              # residuals Q-Q plots
              pdf(sprintf("%sstage2_residualsQQ_norm_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m3, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m3, distribution="norm")
              dev.off()
              pdf(sprintf("%sstage2_residualsQQ_t_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m3, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m3, distribution="t", df=3)
              dev.off()
              jpeg(sprintf("%sstage2_residualsQQ_norm_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m3, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m3, distribution="norm")
              dev.off()
              jpeg(sprintf("%sstage2_residualsQQ_t_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m3, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m3, distribution="t", df=3)
              dev.off()
              pdf(sprintf("%sstage2_residualsQQ_t15_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m3, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m3, distribution="t", df=15)
              dev.off()
              pdf(sprintf("%sstage2_residualsQQ_t10_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m3, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m3, distribution="t", df=10)
              dev.off()
              jpeg(sprintf("%sstage2_residualsQQ_t15_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m3, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m3, distribution="t", df=15)
              dev.off()
              jpeg(sprintf("%sstage2_residualsQQ_t10_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m3, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m3, distribution="t", df=10)
              dev.off()
              violindata <- data.frame(x=rep(sprintf("Window starting on %s", rpy2df$Dates[i]), length(e_st2m3)), y=c(e_st2m3))
              ggplot(violindata, aes(x=x, y=y)) + geom_violin(trim=FALSE) + geom_boxplot(width=0.2) +
                labs(title=sprintf("Residuals - window starting on %s", rpy2df$Dates[i])) + theme_classic()
              ggsave(sprintf("%sstage2_residualsViolin_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m3, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]), width = 7, height = 7, units = "in")
              ggplot(violindata, aes(x=x, y=y)) + geom_violin(trim=FALSE) + geom_boxplot(width=0.2) +
                labs(title=sprintf("Residuals - window starting on %s", rpy2df$Dates[i])) + theme_classic()
              ggsave(sprintf("%sstage2_residualsViolin_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m3, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]), width = 7, height = 7, units = "in")

              pdf(sprintf("%sstage2_residualsVsfitted_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m3, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              plot(e_st2m3, m3_IV[2:length(m3_IV)], main="Residuals vs Fitted Instrumental Variable", xlab="Residuals", ylab="IV", pch=19)
              dev.off()
              jpeg(sprintf("%sstage2_residualsVsfitted_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m3, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              plot(e_st2m3, m3_IV[2:length(m3_IV)], main="Residuals vs Fitted Instrumental Variable", xlab="Residuals", ylab="IV", pch=19)
              dev.off()

              log_print("########## Stage II - M3 ########## ")
              log_print("**********")
              log_print(summary(stage2_m3))
              log_print("**********")
              # LM test for autocorrelation
              lmtest_m3 <- bgtest(stage2_m3)
              log_print(lmtest_m3)
              pm3 <- lmtest_m3$p.val
              # the null hypothesis is that there is no serial correlation of any order up to p
              if (pm3 <= 0.05) {
                log_print("M3 has bad IV")
              } else {
                log_print("M3 has good IV")
                log_print(pm3)
                if (length(which(is.na(stage2_m3$coefficients))) > 0){
                  print("**************** CHECK MODEL 3 REGRESSION *****************")
                }
                coefs <- stage2_m3$coefficients
                pvals <- rep("NA", length(coefs))
                rem <- 0
                for (kkk in 1:length(coefs)){
                  if (is.na(coefs[kkk])){
                     pvals[kkk] <- "NA"
                     rem <- rem + 1
                  }
                  else {
                     pvals[kkk] <- summary(stage2_m3)$coefficients[(kkk-rem), 4]
                  }
                }
                coefs[which(is.na(coefs))] <- -1000
                coefs[pvals > 0.05] <- 0
                if (coefs[3] > 0) {
                  #print(summary(stage2_m3))
                  save_for_aic[3] <- AIC(stage2_m3)
                }
              }
            } else {
              log_print("All coefficients of IV regression were insignificant, change model? - skipping regression with fitted IV")
              log_print(abs(sum(m3_IV)))
            }

            # M4: BERT + VADER, IV regr. BERT
            DIR_out_m4 <- sprintf("%scasestudy1/koyck_results_smooth%s/M4/", DIR, smoothing_win)
            print(DIR_out_m4)
            dir.create(DIR_out_m4)

            # get IV - same IV as M2
            if (abs(sum(m2_IV)) > 1e-16) {

              stage2_m4 <- lm(e_st1[2:length(e_st1)] ~  m2_IV[1:length(m2_IV)-1] + x_B[2:length(x_B)] + x_V[2:length(x_V)])
              if (sum(is.na(stage2_m4$coefficients)) > 0){
                 next
              }
              e_st2m4 <- residuals(stage2_m4)

              # ACF/PACF
              pdf(sprintf("%sstage2_ACF_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m4, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              acf(e_st2m4, main = sprintf("Stage 2 errors - window start: %s", rpy2df$Dates[i]))
              dev.off()
              pdf(sprintf("%sstage2_PACF_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m4, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              pacf(e_st2m4, main = sprintf("Stage 2 errors - window start: %s", rpy2df$Dates[i]))
              dev.off()
              jpeg(sprintf("%sstage2_ACF_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m4, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              acf(e_st2m4, main = sprintf("Stage 2 errors - window start: %s", rpy2df$Dates[i]))
              dev.off()
              jpeg(sprintf("%sstage2_PACF_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m4, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              pacf(e_st2m4, main = sprintf("Stage 2 errors - window start: %s", rpy2df$Dates[i]))
              dev.off()

              # residuals Q-Q plots
              pdf(sprintf("%sstage2_residualsQQ_norm_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m4, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m4, distribution="norm")
              dev.off()
              pdf(sprintf("%sstage2_residualsQQ_t_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m4, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m4, distribution="t", df=3)
              dev.off()
              jpeg(sprintf("%sstage2_residualsQQ_norm_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m4, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m4, distribution="norm")
              dev.off()
              jpeg(sprintf("%sstage2_residualsQQ_t_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m4, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m4, distribution="t", df=3)
              dev.off()
              pdf(sprintf("%sstage2_residualsQQ_t15_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m4, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m4, distribution="t", df=15)
              dev.off()
              pdf(sprintf("%sstage2_residualsQQ_t10_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m4, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m4, distribution="t", df=10)
              dev.off()
              jpeg(sprintf("%sstage2_residualsQQ_t15_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m4, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m4, distribution="t", df=15)
              dev.off()
              jpeg(sprintf("%sstage2_residualsQQ_t10_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m4, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m4, distribution="t", df=10)
              dev.off()
              violindata <- data.frame(x=rep(sprintf("Window starting on %s", rpy2df$Dates[i]), length(e_st2m4)), y=c(e_st2m4))
              ggplot(violindata, aes(x=x, y=y)) + geom_violin(trim=FALSE) + geom_boxplot(width=0.2) +
                labs(title=sprintf("Residuals - window starting on %s", rpy2df$Dates[i])) + theme_classic()
              ggsave(sprintf("%sstage2_residualsViolin_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m4, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]), width = 7, height = 7, units = "in")
              ggplot(violindata, aes(x=x, y=y)) + geom_violin(trim=FALSE) + geom_boxplot(width=0.2) +
                labs(title=sprintf("Residuals - window starting on %s", rpy2df$Dates[i])) + theme_classic()
              ggsave(sprintf("%sstage2_residualsViolin_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m4, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]), width = 7, height = 7, units = "in")


              log_print("########## Stage II - M4 ########## ")
              log_print("**********")
              log_print(summary(stage2_m4))
              log_print("**********")
              # LM test for autocorrelation
              lmtest_m4 <- bgtest(stage2_m4)
              log_print(lmtest_m4)
              pm4 <- lmtest_m4$p.val
              # the null hypothesis is that there is no serial correlation of any order up to p
              if (pm4 <= 0.05) {
                log_print("M4 has bad IV")
              } else {
                log_print("M4 has good IV")
                log_print(pm4)
                if (length(which(is.na(stage2_m4$coefficients))) > 0){
                  print("**************** CHECK MODEL 4 REGRESSION *****************")
                }
                # pvals <- summary(stage2_m4)$coefficients[,4]
                # coefs <- stage2_m4$coefficients
                # coefs[pvals > 0.05] <- 0
                coefs <- stage2_m4$coefficients
                pvals <- rep("NA", length(coefs))
                rem <- 0
                for (kkk in 1:length(coefs)){
                  if (is.na(coefs[kkk])){
                     pvals[kkk] <- "NA"
                     rem <- rem + 1
                  }
                  else {
                     pvals[kkk] <- summary(stage2_m4)$coefficients[(kkk-rem), 4]
                  }
                }
                coefs[which(is.na(coefs))] <- -1000
                coefs[pvals > 0.05] <- 0
                if (coefs[3] > 0 || coefs[4] > 0) {
                  #print(summary(stage2_m4))
                  save_for_aic[4] <- AIC(stage2_m4)
                }
              }
            } else {
              log_print("All coefficients of IV regression were insignificant, change model? - skipping regression with fitted IV")
              log_print(abs(sum(m2_IV)))
            }

            # M5: BERT + VADER, IV regr. VADER
            DIR_out_m5 <- sprintf("%scasestudy1/koyck_results_smooth%s/M5/", DIR, smoothing_win)
            print(DIR_out_m5)
            dir.create(DIR_out_m5)

            # get IV - same IV as M3
            if (abs(sum(m3_IV)) > 1e-16) {
              #print(m3_IV)

              stage2_m5 <- lm(e_st1[2:length(e_st1)] ~  m3_IV[1:length(m3_IV)-1] + x_B[2:length(x_B)] + x_V[2:length(x_V)])
              if (sum(is.na(stage2_m5$coefficients)) > 0) {
                 next
              }
              e_st2m5 <- residuals(stage2_m5)

              # ACF/PACF
              pdf(sprintf("%sstage2_ACF_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m5, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              acf(e_st2m5, main = sprintf("Stage 2 errors - window start: %s", rpy2df$Dates[i]))
              dev.off()
              pdf(sprintf("%sstage2_PACF_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m5, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              pacf(e_st2m5, main = sprintf("Stage 2 errors - window start: %s", rpy2df$Dates[i]))
              dev.off()
              jpeg(sprintf("%sstage2_ACF_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m5, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              acf(e_st2m5, main = sprintf("Stage 2 errors - window start: %s", rpy2df$Dates[i]))
              dev.off()
              jpeg(sprintf("%sstage2_PACF_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m5, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              pacf(e_st2m5, main = sprintf("Stage 2 errors - window start: %s", rpy2df$Dates[i]))
              dev.off()

              # residuals Q-Q plots
              pdf(sprintf("%sstage2_residualsQQ_norm_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m5, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m5, distribution="norm")
              dev.off()
              pdf(sprintf("%sstage2_residualsQQ_t_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m5, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m5, distribution="t", df=3)
              dev.off()
              jpeg(sprintf("%sstage2_residualsQQ_norm_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m5, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m5, distribution="norm")
              dev.off()
              jpeg(sprintf("%sstage2_residualsQQ_t_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m5, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m5, distribution="t", df=3)
              dev.off()
              pdf(sprintf("%sstage2_residualsQQ_t15_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m5, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m5, distribution="t", df=15)
              dev.off()
              pdf(sprintf("%sstage2_residualsQQ_t10_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m5, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m5, distribution="t", df=10)
              dev.off()
              jpeg(sprintf("%sstage2_residualsQQ_t15_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m5, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m5, distribution="t", df=15)
              dev.off()
              jpeg(sprintf("%sstage2_residualsQQ_t10_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m5, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m5, distribution="t", df=10)
              dev.off()
              violindata <- data.frame(x=rep(sprintf("Window starting on %s", rpy2df$Dates[i]), length(e_st2m5)), y=c(e_st2m5))
              ggplot(violindata, aes(x=x, y=y)) + geom_violin(trim=FALSE) + geom_boxplot(width=0.2) +
                labs(title=sprintf("Residuals - window starting on %s", rpy2df$Dates[i])) + theme_classic()
              ggsave(sprintf("%sstage2_residualsViolin_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m5, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]), width = 7, height = 7, units = "in")
              ggplot(violindata, aes(x=x, y=y)) + geom_violin(trim=FALSE) + geom_boxplot(width=0.2) +
                labs(title=sprintf("Residuals - window starting on %s", rpy2df$Dates[i])) + theme_classic()
              ggsave(sprintf("%sstage2_residualsViolin_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m5, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]), width = 7, height = 7, units = "in")


              log_print("########## Stage II - M5 ########## ")
              log_print("**********")
              log_print(summary(stage2_m5))
              log_print("**********")
              # LM test for autocorrelation
              lmtest_m5 <- bgtest(stage2_m5)
              log_print(lmtest_m5)
              pm5 <- lmtest_m5$p.val
              # the null hypothesis is that there is no serial correlation of any order up to p
              if (pm5 <= 0.05) {
                log_print("M5 has bad IV")
              } else {
                log_print("M5 has good IV")
                log_print(pm5)
                if (length(which(is.na(stage2_m5$coefficients))) > 0){
                  print("**************** CHECK MODEL 5 REGRESSION *****************")
                }
                coefs <- stage2_m5$coefficients
                pvals <- rep("NA", length(coefs))
                rem <- 0
                for (kkk in 1:length(coefs)){
                  if (is.na(coefs[kkk])){
                     pvals[kkk] <- "NA"
                     rem <- rem + 1
                  }
                  else {
                     pvals[kkk] <- summary(stage2_m5)$coefficients[(kkk-rem), 4]
                  }
                }
                coefs[which(is.na(coefs))] <- -1000
                coefs[pvals > 0.05] <- 0
                if (coefs[3] > 0 || coefs[4] > 0) {
                  #print(summary(stage2_m5))
                  save_for_aic[5] <- AIC(stage2_m5)
                }
              }
            } else {
              log_print("All coefficients of IV regression were insignificant, change model? - skipping regression with fitted IV")
              log_print(abs(sum(m3_IV)))
            }

            # M6: BERT + VADER, IV regr. BERT + VADER
            DIR_out_m6 <- sprintf("%scasestudy1/koyck_results_smooth%s/M6/", DIR, smoothing_win)
            print(DIR_out_m6)
            dir.create(DIR_out_m6)

            stage2d <- lm(e_st1 ~ x_B + x_V)
            log_print("########## Stage II - M6 IV fit ########## ")
            log_print("**********")
            log_print(summary(stage2d))
            log_print("**********")
            # get IV
            m6_IV <- fit_reduced_lm(stage2d)

            if (abs(sum(m6_IV)) > 1e-16) {
              #print(m6_IV)

              stage2_m6 <- lm(e_st1[2:length(e_st1)] ~  m6_IV[1:length(m6_IV)-1] + x_B[2:length(x_B)] + x_V[2:length(x_V)])

              if (sum(is.na(stage2_m6$coefficients)) > 0){
                 next
              }

              e_st2m6 <- residuals(stage2_m6)

              # ACF/PACF
              pdf(sprintf("%sstage2_ACF_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m6, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              acf(e_st2m6, main = sprintf("Stage 2 errors - window start: %s", rpy2df$Dates[i]))
              dev.off()
              pdf(sprintf("%sstage2_PACF_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m6, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              pacf(e_st2m6, main = sprintf("Stage 2 errors - window start: %s", rpy2df$Dates[i]))
              dev.off()
              jpeg(sprintf("%sstage2_ACF_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m6, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              acf(e_st2m6, main = sprintf("Stage 2 errors - window start: %s", rpy2df$Dates[i]))
              dev.off()
              jpeg(sprintf("%sstage2_PACF_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m6, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              pacf(e_st2m6, main = sprintf("Stage 2 errors - window start: %s", rpy2df$Dates[i]))
              dev.off()

              # residuals Q-Q plots
              pdf(sprintf("%sstage2_residualsQQ_norm_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m6, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m6, distribution="norm")
              dev.off()
              pdf(sprintf("%sstage2_residualsQQ_t_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m6, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m6, distribution="t", df=3)
              dev.off()
              jpeg(sprintf("%sstage2_residualsQQ_norm_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m6, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m6, distribution="norm")
              dev.off()
              jpeg(sprintf("%sstage2_residualsQQ_t_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m6, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m6, distribution="t", df=3)
              dev.off()
              pdf(sprintf("%sstage2_residualsQQ_t15_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m6, entropy_type,
                        smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m6, distribution="t", df=15)
              dev.off()
              pdf(sprintf("%sstage2_residualsQQ_t10_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m6, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m6, distribution="t", df=10)
              dev.off()
              jpeg(sprintf("%sstage2_residualsQQ_t15_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m6, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m6, distribution="t", df=15)
              dev.off()
              jpeg(sprintf("%sstage2_residualsQQ_t10_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m6, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              qqPlot(e_st2m6, distribution="t", df=10)
              dev.off()
              violindata <- data.frame(x=rep(sprintf("Window starting on %s", rpy2df$Dates[i]), length(e_st2m6)), y=c(e_st2m6))
              ggplot(violindata, aes(x=x, y=y)) + geom_violin(trim=FALSE) + geom_boxplot(width=0.2) +
                labs(title=sprintf("Residuals - window starting on %s", rpy2df$Dates[i])) + theme_classic()
              ggsave(sprintf("%sstage2_residualsViolin_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m6, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]), width = 7, height = 7, units = "in")
              ggplot(violindata, aes(x=x, y=y)) + geom_violin(trim=FALSE) + geom_boxplot(width=0.2) +
                labs(title=sprintf("Residuals - window starting on %s", rpy2df$Dates[i])) + theme_classic()
              ggsave(sprintf("%sstage2_residualsViolin_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m6, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]), width = 7, height = 7, units = "in")

              pdf(sprintf("%sstage2_residualsVsfitted_%s_median_smooth%s_window%s_%s.pdf", DIR_out_m6, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              plot(e_st2m6, m6_IV[2:length(m6_IV)], main="Residuals vs Fitted Instrumental Variable", xlab="Residuals", ylab="IV", pch=19)
              dev.off()
              jpeg(sprintf("%sstage2_residualsVsfitted_%s_median_smooth%s_window%s_%s.jpeg", DIR_out_m6, entropy_type,
                          smoothing_win, window_size, rpy2df$Dates[i]))
              plot(e_st2m6, m6_IV[2:length(m6_IV)], main="Residuals vs Fitted Instrumental Variable", xlab="Residuals", ylab="IV", pch=19)
              dev.off()

              log_print("########## Stage II - M6 ########## ")
              log_print("**********")
              log_print(summary(stage2_m6))
              log_print("**********")
              # LM test for autocorrelation
              lmtest_m6 <- bgtest(stage2_m6)
              log_print(lmtest_m6)
              pm6 <- lmtest_m6$p.val
              # the null hypothesis is that there is no serial correlation of any order up to p
              if (pm6 <= 0.05) {
                log_print("M6 has bad IV")
              } else {
                log_print("M6 has good IV")
                log_print(pm6)
                if (length(which(is.na(stage2_m6$coefficients))) > 0){
                  print("**************** CHECK MODEL 6 REGRESSION *****************")
                }
                coefs <- stage2_m6$coefficients
                pvals <- rep("NA", length(coefs))
                rem <- 0
                for (kkk in 1:length(coefs)){
                  if (is.na(coefs[kkk])){
                     pvals[kkk] <- "NA"
                     rem <- rem + 1
                  }
                  else {
                     pvals[kkk] <- summary(stage2_m6)$coefficients[(kkk-rem), 4]
                  }
                }
                coefs[which(is.na(coefs))] <- -1000
                coefs[pvals > 0.05] <- 0
                if (coefs[3] > 0 || coefs[4] > 0) {
                  # print(summary(stage2_m6))
                  save_for_aic[6] <- AIC(stage2_m6)
                }
              }
            } else {
              log_print("All coefficients of IV regression were insignificant,
                            change model? - skipping regression with fitted IV")
              log_print(abs(sum(m6_IV)))
            }

            # get best (AIC) model for current window
            best_model <- which(save_for_aic == min(save_for_aic[!is.na(save_for_aic)]))

            if (length(best_model) > 1) {
                # choose the first, since all models are equally good
                best_model <- best_model[1]
            }
            if (best_model > 1) {
              best_m <- get(paste("stage2_m", best_model, sep=""))
              best_IV_regr[k] <- paste("stage2_m", best_model, sep="")
              # get regression coefficients and confidence intervals
              best_coef <- summary(best_m)$coefficients[,1]
              best_coef_pval <- summary(best_m)$coefficients[,4]

              # Cochrane-Orcutt adjusted errors
              best_coef_ci <- summary(best_m)$coefficients[,2]
              days_win[k] <- rpy2df$Dates[i]

              if (length(which(is.na(best_m$coefficients))) > 0){
                print("**************** CHECK BEST MODEL REGRESSION *****************")
                print(which(is.na(best_m$coefficients)))
                best_coef <- rep(-1000, length(best_m$coefficients))
                best_coef_pval <- rep(-1000, length(best_m$coefficients))
                best_coef_ci <- rep(-1000, length(best_m$coefficients))
                for (jj in seq(length(best_m$coefficients))) {
                  if (jj == which(is.na(best_m$coefficients))) {
                    print(best_m)
                    best_coef[jj] <- "NA" #best_m$coefficients[jj, 1]
                    best_coef_pval[jj] <- "NA" #best_m$coefficients[jj, 4]
                    best_coef_ci[jj] <- "NA" #best_m$coefficients[jj, 2]
                  }
                }
              }
              if (best_model == 2) {
                intercepts[k] <- best_coef[1]/(1 - best_coef[2])
                phis[k] <- best_coef[2]
                betas1[k] <- best_coef[3]

                rseA[k] <- best_coef_ci[1]/(1 - best_coef[2]^2)
                rsePhi[k] <- (1.96/sqrt(window_size))*best_coef_ci[2]/((1 - best_coef[2]^2)*(1 + best_coef[2]^2))
                rseB1[k] <- (1.96/sqrt(window_size))*best_coef_ci[3]/((1 - best_coef[2]^2)*(1 + best_coef[2]^2))

                pvalA[k] <- best_coef_pval[1]
                pvalPhi[k] <- best_coef_pval[2]
                pvalB1[k] <- best_coef_pval[3]

              } else if (best_model == 3) {
                intercepts[k] <- best_coef[1]/(1 - best_coef[2])
                phis[k] <- best_coef[2]
                betas2[k] <- best_coef[3]

                rseA[k] <- best_coef_ci[1]/(1 - best_coef[2]^2)
                rsePhi[k] <- (1.96/sqrt(window_size))*best_coef_ci[2]/((1 - best_coef[2]^2)*(1 + best_coef[2]^2))
                rseB2[k] <- (1.96/sqrt(window_size))*best_coef_ci[3]/((1 - best_coef[2]^2)*(1 + best_coef[2]^2))

                pvalA[k] <- best_coef_pval[1]
                pvalPhi[k] <- best_coef_pval[2]
                pvalB2[k] <- best_coef_pval[3]

              } else if (best_model > 3 && best_model <= 6) {
                intercepts[k] <- best_coef[1]/(1 - best_coef[2])
                phis[k] <- best_coef[2]
                betas1[k] <- best_coef[3]
                betas2[k] <- best_coef[4]

                rseA[k] <- best_coef_ci[1]/(1 - best_coef[2]^2)
                rsePhi[k] <- (1.96/sqrt(window_size))*best_coef_ci[2]/((1 - best_coef[2]^2)*(1 + best_coef[2]^2))
                rseB1[k] <- (1.96/sqrt(window_size))*best_coef_ci[3]/((1 - best_coef[2]^2)*(1 + best_coef[2]^2))
                rseB2[k] <- (1.96/sqrt(window_size))*best_coef_ci[4]/((1 - best_coef[2]^2)*(1 + best_coef[2]^2))

                pvalA[k] <- best_coef_pval[1]
                pvalPhi[k] <- best_coef_pval[2]
                pvalB1[k] <- best_coef_pval[3]
                pvalB2[k] <- best_coef_pval[4]
              }


              if (pvalB1[k] !=-1000 && pvalB1[k] <= 0.001) {
                 stage2_significance_BERT[k] <- "***"
              }
              else if (pvalB1[k] !=-1000 && pvalB1[k] <= 0.01) {
                 stage2_significance_BERT[k] <- "**"
              }
              else if (pvalB1[k] !=-1000 && pvalB1[k] <= 0.05) {
                 stage2_significance_BERT[k] <- "*"
              }
              else {
                 stage2_significance_BERT[k] <- "."
              }

              if (pvalB2[k] !=-1000 && pvalB2[k] <= 0.001) {
                 stage2_significance_VADER[k] <- "***"
              }
              else if (pvalB2[k] !=-1000 && pvalB2[k] <= 0.01) {
                 stage2_significance_VADER[k] <- "**"
              }
              else if (pvalB2[k] !=-1000 && pvalB2[k] <= 0.05) {
                 stage2_significance_VADER[k] <- "*"
              }
              else {
                 stage2_significance_VADER[k] <- "."
              }

            }

            k  <- k + 1
            if (i + stepsize + window_size > length(response)){
              print("Next window beyond time-series end - stop")
              break
            }
            log_print("*****************************************************************************")

          }

          log_close()

          if (length(phis[phis!=-1000]) > 0) {
            dataout <- data.frame("best_regr_model"=best_IV_regr,
                                  "alpha"=intercepts,
                                  "betaBERT"=betas1,
                                  "betaVADER"=betas2,
                                  "phi"=phis,
                                  "rseA"=rseA,
                                  "pvalA"=pvalA,
                                  "rsePhi"=rsePhi,
                                  "pvalPhi"=pvalPhi,
                                  "rseBBERT"=rseB1,
                                  "pvalBBERT"=pvalB1,
                                  "rseBVADER"=rseB2,
                                  "pvalBVADER"=pvalB2,
                                  "dates"=days_win)

            data_test_stage2 <- data.frame(window=days_win, stage2_significance_BERT=stage2_significance_BERT,
                                           stage2_BERT=betas1, errorBERT=rseB1,
                                           stage2_significance_VADER=stage2_significance_VADER, stage2_VADER=betas2,
                                           errorVADER=rseB2, best_model=best_IV_regr)
            write.csv(data_test_stage2, sprintf("%sstage2_test_significance_%s_median_smooth%s_window%s.csv",
                                                DIR_out, entropy_type, smoothing_win, window_size),
                                                row.names = FALSE)
            save(dataout, file=sprintf("%s%s_koyckdlm_bert_vader_ivreg_regression_median_smooth%s.rdata",
                                     DIR_out, entropy_type, smoothing_win))
          }
         stage1_failed <- list(stage1_problematic_windows)
         save(stage1_failed, file=sprintf("%s%s_stage1_AR_failed_windows_postFit.rdata",
                            DIR_out_st1, entropy_type))
     }
}
