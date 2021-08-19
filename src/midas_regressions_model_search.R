#################################################################################

# Ioannis Chalkiadakis
# August, 2021

# Resilient ML

#################################################################################

# install.packages(c("midasr", "roll", "car"))
library(midasr)
library(roll)
library(car)


nealmon_gradient <- function(p, d, m) {
     i <- 1:d
     pl <- poly(i, degree = length(p) - 1, raw = TRUE)
     eplc <- exp(pl %*% p[-1])[, , drop = TRUE]
     ds <- colSums(pl * eplc)
     s <- sum(eplc)
     cbind(eplc / s, p[1] * (pl * eplc / s - eplc %*% t(ds) / s^2))
}


# Load data
fname_sent <- "./data/volume_joint_weighted_per_asset_cubic_interpolated_median_smooth7days_timeseries.csv"
btc_abs_entropy <- read.csv(fname_sent)
btc_sent_entropy <- btc_abs_entropy$entropy_abs_strength_BTC_nonsmooth
btc_sent_pos_entropy <- btc_abs_entropy$entropy_positive_BTC_nonsmooth
btc_sent_neg_entropy <- btc_abs_entropy$entropy_negative_BTC_nonsmooth

fname_mining <- "./data/mining_df_btc.csv"
mining_df_btc <- read.csv(fname_mining)
x_mining <- as.numeric(mining_df_btc$hashrate)

fname_btc_close <- "./data/btc_close.csv"
btc_close <- read.csv(fname_btc_close)

DIR_out <- "/tmp/"
print(DIR_out)
dir.create(DIR_out)

# Max autoregrssive lag - days
max_p <- 5
# Max hashrate lag - months
max_p_hashrate <- 6
# Max btc_close lag - days
max_p_btcclose <- 5
# Best hashrate Almon k
max_k1 <- 5
# Best BTC close Almon k
max_k2 <- 5

sentiment <- rep("NA", 20000)
autoregressive_lag <- rep("NA", 20000)
hashrate_lag <- rep("NA", 20000)
btcclose_lag <- rep("NA", 20000)
differentiate <- rep("NA", 20000)
smoothed <- rep("NA", 20000)
converged_alg <- rep("NA", 20000)
optimisation <- rep("NA", 20000)
hashrate_almonlag <- rep("NA", 20000)
btcclose_almonlag <- rep("NA", 20000)
aics <- rep("NA", 20000)
mses <- rep("NA", 20000)

jjj <- 1

for (smooth in list(FALSE, TRUE)){
  for (diff in list(FALSE, TRUE)){
    if (diff){
      smooth <- FALSE
    }
    for (ytype in list("total", "neg", "pos")){
        if (ytype == "total") {
           y <- btc_sent_entropy[7:length(btc_sent_entropy)]
        }
        else if (ytype == "neg"){
          y <- btc_sent_neg_entropy[7:length(btc_sent_neg_entropy)]
        }
        else {
          y <- btc_sent_pos_entropy[7:length(btc_sent_pos_entropy)]
        }
        # Detrend response
        y <- (y - mean(y))
        lags_comb <- list(list(max_p:max_p, max_p_hashrate:max_p_hashrate, 1:max_p_btcclose),
                          list(max_p:max_p, 1:max_p_hashrate, max_p_btcclose:max_p_btcclose),
                          list(1:max_p, max_p_hashrate:max_p_hashrate, max_p_btcclose:max_p_btcclose))
        for (lc in lags_comb){
            for (yl in lc[1]){
               for (xl in lc[2]){
                  for (zl in lc[3]){
                    for (k1 in 1:max_k1) {
                      for (k2 in 1:max_k2) {
                          # Standardise covariates
                          x <- (x_mining - mean(x_mining))/sd(x_mining)
                          z <- (btc_close$close - mean(btc_close$close))/sd(btc_close$close)
                          if (diff){
                            z <- diff(z)
                            z <- z[24:length(z)]
                            x <- diff(x)
                            y <- y[2:length(y)]
                          }
                          else{
                            if ((smooth) && (!diff)) {
                              # 1-week smoothing
                              x <- roll_median(x, width=7)
                              x <- x[8:length(x)]
                              z <- roll_median(z, width=7)
                              z <- z[(24*7+1):length(z)]
                              y <- y[8:length(y)]
                            }
                          }
                          converged <- 1

                          if (length(yl) > 1){
                            for (yll in yl){
                              print(yll)
                              print(xl)
                              print(zl)
                              converged <- tryCatch({
                                      eqr <- midas_r(y ~  mls(y, 1:yll, 1) + mls(z, 1:(zl*24), 24, nealmon) + mls(x, 1:(xl*30), 1, nealmon),
                                      data=list(y=y, z=z, x=x), Ofunction = "optim", method="Nelder-Mead",
                                      start=list(z=rep(-0.1, k2+1), x=rep(-0.1, k1+1)))
                                      summary(eqr)
                                      alg <- "Nelder-Mead"
                                      converged <- eqr$opt$convergence
                                      }, error=function(n){
                                         return (1)
                                      })
                              if (converged == 1){
                                converged <- tryCatch({
                                        eqr <- midas_r(y ~  mls(y, 1:yll, 1) + mls(z, 1:(zl*24), 24, nealmon) + mls(x, 1:(xl*30), 1, nealmon),
                                                  data=list(y=y, z=z, x=x), Ofunction = "optim", method="BFGS",
                                                  start=list(z=rep(-0.1, k2+1), x=rep(-0.1, k1+1)))
                                        summary(eqr)
                                        alg <- "BFGS"
                                        converged <- eqr$opt$convergence
                                        }, error=function(n){
                                           return (1)
                                        })
                              }
                              if (converged == 1){
                                converged <- tryCatch({
                                      eqr <- midas_r(y ~  mls(y, 1:yl, 1) + mls(z, 1:(zll*24), 24, nealmon) + mls(x, 1:(xl*30), 1, nealmon),
                                                data=list(y=y, z=z, x=x), Ofunction = "optim", method="BFGS",
                                                start=list(z=rep(-0.1, k2+1), x=rep(-0.1, k1+1)),
                                                weight_gradients=list(nealmon=nealmon_gradient))
                                      summary(eqr)
                                      alg <- "BFGS_analytic"
                                      print(alg)
                                      print(eqr$opt$convergence)
                                      converged <- eqr$opt$convergence
                                      }, error=function(n){
                                         return (1)
                                      })
                              }
                              if (converged == 1){
                                converged <- tryCatch({
                                      eqr <- update(eqr, Ofunction="nls")
                                      summary(eqr)
                                      alg <- "NLS_BFGS_start"
                                      print(alg)
                                      print(eqr$convergence)
                                      converged <- eqr$convergence
                                      }, error=function(n){
                                         return (1)
                                      })
                              }
                              if (converged == 0){
                                  print("Converged")
                                  print(converged)
                                  if (!file.exists(sprintf("%smidas_fit_%s_Residuals_norm_autoregrLag_%s_HRLag_%s_BTCLag_%s_diff_%s_smooth_%s_converge_%s_%s_HR_almon_%s_BTC_almon_%s.jpeg",
                                                  DIR_out, ytype, yll, xl, zl, diff, smooth, converged, alg, k1, k2))){
                                      jpeg(sprintf("%smidas_fit_%s_Residuals_norm_autoregrLag_%s_HRLag_%s_BTCLag_%s_diff_%s_smooth_%s_converge_%s_%s_HR_almon_%s_BTC_almon_%s.jpeg",
                                                      DIR_out, ytype, yll, xl, zl, diff, smooth, converged, alg, k1, k2))
                                      plot(residuals(eqr), type="l")
                                      dev.off()
                                      jpeg(sprintf("%smidas_fit_%s_QQ_norm_autoregrLag_%s_HRLag_%s_BTCLag_%s_diff_%s_smooth_%s_converge_%s_%s_HR_almon_%s_BTC_almon_%s.jpeg",
                                                      DIR_out, ytype, yll, xl, zl, diff, smooth, converged, alg, k1, k2))
                                      qqPlot(residuals(eqr), distribution="norm")
                                      dev.off()
                                  }
                                  aic <- AIC(eqr)
                                  mse <- mean((fitted(eqr) - y[(1+xl*30):length(y)])^2)
                                  aics[jjj] <- aic
                                  mses[jjj] <- mse
                                  autoregressive_lag[jjj] <- yll
                                  hashrate_lag[jjj] <- xl
                                  btcclose_lag[jjj] <- zl
                              }

                            }
                          }
                          else if (length(xl) > 1){
                            for (xll in xl){
                              print(yl)
                              print(xll)
                              print(zl)
                              converged <- tryCatch({
                                        eqr <- midas_r(y ~  mls(y, 1:yl, 1) + mls(z, 1:(zl*24), 24, nealmon) + mls(x, 1:(xll*30), 1, nealmon),
                                                  data=list(y=y, z=z, x=x), Ofunction = "optim", method="Nelder-Mead",
                                                  start=list(z=rep(-0.1, k2+1), x=rep(-0.1, k1+1)))
                                        summary(eqr)
                                        alg <- "Nelder-Mead"
                                        converged <- eqr$opt$convergence
                                      }, error=function(n){
                                           return (1)
                                      })
                              if (converged == 1){
                                converged <- tryCatch({
                                            eqr <- midas_r(y ~  mls(y, 1:yl, 1) + mls(z, 1:(zl*24), 24, nealmon) + mls(x, 1:(xll*30), 1, nealmon),
                                                      data=list(y=y, z=z, x=x), Ofunction = "optim", method="BFGS",
                                                      start=list(z=rep(-0.1, k2+1), x=rep(-0.1, k1+1)))
                                            summary(eqr)
                                            alg <- "BFGS"
                                            converged <- eqr$opt$convergence
                                            }, error=function(n){
                                               return (1)
                                            })
                              }
                              if (converged == 1){
                                converged <- tryCatch({
                                      eqr <- midas_r(y ~  mls(y, 1:yl, 1) + mls(z, 1:(zll*24), 24, nealmon) + mls(x, 1:(xl*30), 1, nealmon),
                                                data=list(y=y, z=z, x=x), Ofunction = "optim", method="BFGS",
                                                start=list(z=rep(-0.1, k2+1), x=rep(-0.1, k1+1)),
                                                weight_gradients=list(nealmon=nealmon_gradient))
                                      summary(eqr)
                                      alg <- "BFGS_analytic"
                                      print(alg)
                                      print(eqr$opt$convergence)
                                      converged <- eqr$opt$convergence
                                      }, error=function(n){
                                         return (1)
                                      })
                              }
                              if (converged == 1){
                                converged <- tryCatch({
                                      eqr <- update(eqr, Ofunction="nls")
                                      summary(eqr)
                                      alg <- "NLS_BFGS_start"
                                      print(alg)
                                      print(eqr$convergence)
                                      converged <- eqr$convergence
                                      }, error=function(n){
                                         return (1)
                                      })
                              }
                              if (converged == 0){
                                  print("Converged")
                                  print(converged)
                                  if (!file.exists(sprintf("%smidas_fit_%s_Residuals_norm_autoregrLag_%s_HRLag_%s_BTCLag_%s_diff_%s_smooth_%s_converge_%s_%s_HR_almon_%s_BTC_almon_%s.jpeg",
                                                  DIR_out, ytype, yl, xll, zl, diff, smooth, converged, alg, k1, k2))){
                                      jpeg(sprintf("%smidas_fit_%s_Residuals_norm_autoregrLag_%s_HRLag_%s_BTCLag_%s_diff_%s_smooth_%s_converge_%s_%s_HR_almon_%s_BTC_almon_%s.jpeg",
                                                      DIR_out, ytype, yl, xll, zl, diff, smooth, converged, alg, k1, k2))
                                      plot(residuals(eqr), type="l")
                                      dev.off()
                                      jpeg(sprintf("%smidas_fit_%s_QQ_norm_autoregrLag_%s_HRLag_%s_BTCLag_%s_diff_%s_smooth_%s_converge_%s_%s_HR_almon_%s_BTC_almon_%s.jpeg",
                                                      DIR_out, ytype, yl, xll, zl, diff, smooth, converged, alg, k1, k2))
                                      qqPlot(residuals(eqr), distribution="norm")
                                      dev.off()
                                  }
                                  aic <- AIC(eqr)
                                  mse <- mean((fitted(eqr) - y[(1+xll*30):length(y)])^2)
                                  aics[jjj] <- aic
                                  mses[jjj] <- mse
                                  autoregressive_lag[jjj] <- yl
                                  hashrate_lag[jjj] <- xll
                                  btcclose_lag[jjj] <- zl
                              }
                            }
                          }
                          else {
                            for (zll in zl){
                              print(yl)
                              print(xl)
                              print(zll)
                              converged <- tryCatch({
                                      eqr <- midas_r(y ~  mls(y, 1:yl, 1) + mls(z, 1:(zll*24), 24, nealmon) + mls(x, 1:(xl*30), 1, nealmon),
                                                data=list(y=y, z=z, x=x), Ofunction = "optim", method="Nelder-Mead",
                                                start=list(z=rep(-0.1, k2+1), x=rep(-0.1, k1+1)))
                                      summary(eqr)
                                      alg <- "Nelder-Mead"
                                      converged <- eqr$opt$convergence
                                    }, error=function(n){
                                         return (1)
                                    })
                              if (converged == 1){
                                converged <- tryCatch({
                                      eqr <- midas_r(y ~  mls(y, 1:yl, 1) + mls(z, 1:(zll*24), 24, nealmon) + mls(x, 1:(xl*30), 1, nealmon),
                                                data=list(y=y, z=z, x=x), Ofunction = "optim", method="BFGS",
                                                start=list(z=rep(-0.1, k2+1), x=rep(-0.1, k1+1)))
                                      summary(eqr)
                                      alg <- "BFGS"
                                      converged <- eqr$opt$convergence
                                      }, error=function(n){
                                         return (1)
                                      })
                              }
                              if (converged == 1){
                                converged <- tryCatch({
                                      eqr <- midas_r(y ~  mls(y, 1:yl, 1) + mls(z, 1:(zll*24), 24, nealmon) + mls(x, 1:(xl*30), 1, nealmon),
                                                data=list(y=y, z=z, x=x), Ofunction = "optim", method="BFGS",
                                                start=list(z=rep(-0.1, k2+1), x=rep(-0.1, k1+1)),
                                                weight_gradients=list(nealmon=nealmon_gradient))
                                      summary(eqr)
                                      alg <- "BFGS_analytic"
                                      print(alg)
                                      print(eqr$opt$convergence)
                                      converged <- eqr$opt$convergence
                                      }, error=function(n){
                                         return (1)
                                      })
                              }
                              if (converged == 1){
                                converged <- tryCatch({
                                      eqr <- update(eqr, Ofunction="nls")
                                      summary(eqr)
                                      alg <- "NLS_BFGS_start"
                                      print(alg)
                                      print(eqr$convergence)
                                      converged <- eqr$convergence
                                      }, error=function(n){
                                         return (1)
                                      })
                              }
                              if (converged == 0){
                                  print("Converged")
                                  print(converged)
                                  if (!file.exists(sprintf("%smidas_fit_%s_Residuals_norm_autoregrLag_%s_HRLag_%s_BTCLag_%s_diff_%s_smooth_%s_converge_%s_%s_HR_almon_%s_BTC_almon_%s.jpeg",
                                                  DIR_out, ytype, yl, xl, zll, diff, smooth, converged, alg, k1, k2))){
                                    jpeg(sprintf("%smidas_fit_%s_Residuals_norm_autoregrLag_%s_HRLag_%s_BTCLag_%s_diff_%s_smooth_%s_converge_%s_%s_HR_almon_%s_BTC_almon_%s.jpeg",
                                                  DIR_out, ytype, yl, xl, zll, diff, smooth, converged, alg, k1, k2))
                                    plot(residuals(eqr), type="l")
                                    dev.off()
                                    jpeg(sprintf("%smidas_fit_%s_QQ_norm_autoregrLag_%s_HRLag_%s_BTCLag_%s_diff_%s_smooth_%s_converge_%s_%s_HR_almon_%s_BTC_almon_%s.jpeg",
                                                  DIR_out, ytype, yl, xl, zll, diff, smooth, converged, alg, k1, k2))
                                    qqPlot(residuals(eqr), distribution="norm")
                                    dev.off()
                                  }
                                  aic <- AIC(eqr)
                                  mse <- mean((fitted(eqr) - y[(1+xl*30):length(y)])^2)
                                  aics[jjj] <- aic
                                  mses[jjj] <- mse
                                  autoregressive_lag[jjj] <- yl
                                  hashrate_lag[jjj] <- xl
                                  btcclose_lag[jjj] <- zll
                                }
                            }
                          }

                          if (converged == 0){
                            sentiment[jjj] <- ytype
                            differentiate[jjj] <- diff
                            smoothed[jjj] <- smooth
                            converged_alg[jjj] <- converged
                            optimisation[jjj] <- alg
                            hashrate_almonlag[jjj] <- k1
                            btcclose_almonlag[jjj] <- k2
                            jjj <- jjj + 1
                          }
                      }
                    }
               }
            }
        }
    }
  }
}
}
data_out <- data.frame(sentiment=sentiment,
                       autoregressive_lag=autoregressive_lag,
                       hashrate_lag=hashrate_lag,
                       btcclose_lag=btcclose_lag,
                       differentiate=differentiate,
                       smoothed=smoothed,
                       converged_alg=converged_alg,
                       optimisation=optimisation,
                       hashrate_almonlag=hashrate_almonlag,
                       btcclose_almonlag=btcclose_almonlag,
                       aic=aics,
                       mse=mses)
write.csv(data_out, sprintf("%smidas_fits.csv", DIR_out), row.names = FALSE)






