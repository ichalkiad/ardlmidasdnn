#################################################################################

# Ioannis Chalkiadakis
# August, 2021

# Resilient ML

#################################################################################

library("midasr")
library("roll")
library("car")
library("pracma")


nealmon_gradient <- function(p, d, m) {
     i <- 1:d
     pl <- poly(i, degree = length(p) - 1, raw = TRUE)
     eplc <- exp(pl %*% p[-1])[, , drop = TRUE]
     ds <- colSums(pl * eplc)
     s <- sum(eplc)
     cbind(eplc / s, p[1] * (pl * eplc / s - eplc %*% t(ds) / s^2))
}


star_significance <- function(m) {

  if (m < 0.001){
    star <- "***"
  }
  else if (m < 0.01){
    star <- "**"
  }
  else if (m < 0.05){
    star <- "**"
  }
  else if (m < 0.1){
    star <- "."
  }
  else{
    star <- "NA"
  }
}


DIR <- "/tmp/"

# Load data
fname_sent <- "./data/volume_joint_weighted_per_asset_cubic_interpolated_median_smooth7days_timeseries.csv"
btc_abs_entropy <- read.csv(fname_sent)

# sentiment response
btc_sent_entropy <- btc_abs_entropy$entropy_abs_strength_BTC_nonsmooth

iv <- btc_abs_entropy$entropy_abs_strength_BTC - (btc_abs_entropy$VADER_BTC + btc_abs_entropy$BERT_BTC)/2

# technology covariate
fname_mining <- "./data/mining_df_btc.csv"
mining_df_btc <- read.csv(fname_mining)
x_mining <- as.numeric(mining_df_btc$hashrate)

# price covariate
fname_btc_close <- "./data/btc_close.csv"
btc_close <- read.csv(fname_btc_close)
# zzdate <- as.POSIXct(btc_close$date,format="%Y-%m-%d %H:%M:%S", tz = "UTC")

# drop fist week to match dates in covariates
y_init <- btc_sent_entropy[7:length(btc_sent_entropy)]
dates_init <- btc_abs_entropy$Dates[7:length(btc_sent_entropy)]
# Detrend response
y_init <- (y_init - mean(y_init))
# Standardise covariates
x_init <- (x_mining - mean(x_mining))/sd(x_mining)
z_init <- (btc_close$close - mean(btc_close$close))/sd(btc_close$close)

configurations <- list(
                        list(ylags=5, zlags=5, xlags=6, boxcox=FALSE, boxcoxL=0, hashrate_almonlag=1, btcclose_almonlag=1, smoothed=TRUE),
                        list(ylags=5, zlags=5, xlags=6, boxcox=FALSE, boxcoxL=0, hashrate_almonlag=1, btcclose_almonlag=2, smoothed=TRUE)
                      )
stepsize <- 30

for (config in configurations) {
    print(config)
    # Set up fitting paramenters according to model selection results
    ylags <- config$ylags
    zlags <- config$zlags
    xlags <- config$xlags
    hashrate_almonlag <- config$hashrate_almonlag
    btcclose_almonlag <- config$btcclose_almonlag
    smoothed <- config$smoothed
    boxCox <- config$boxcox
    boxCox_lamda <- config$boxcoxL
    print(ylags)
    print(zlags)
    print(xlags)
    print(hashrate_almonlag)
    print(btcclose_almonlag)
    print(smoothed)
    print(boxCox)
    print(boxCox_lamda)

    if (smoothed) {
      # 1-week smoothing
      x <- roll_median(x_init, width=7)
      # drop NAs at start
      x <- x_init[8:length(x_init)]
      z <- roll_median(z_init, width=7)
      # drop NAs and also all up to 1 week at start to align with y and x
      # *6 so that we set up the iteration according to i in the loop below - i starts at 1
      z <- z[(24*6+1):length(z_init)]
      zdates <- btc_close$date[(24*6+1):length(btc_close$close)]
      y <- y_init[8:length(y_init)]
      dates <- dates_init[8:length(dates_init)]
      # ensure response/covariate dates are aligned
      stopifnot(dates==mining_df_btc$date[8:length(mining_df_btc$date)])
      mining_dates <- mining_df_btc$date[8:length(mining_df_btc$date)]
    } else{
        x <- x_init
        z <- z_init
        zdates <- btc_close$date
        y <- y_init
        dates <- dates_init
        # ensure response/covariate dates are aligned
        stopifnot(dates==mining_df_btc$date)
        mining_dates <- mining_df_btc$date
    }

    if (btcclose_almonlag==1){
      winsizes <- c(1260)
    } else {
      winsizes <- c(1080)
    }

    for (window_size in winsizes){

          if (boxCox) {
            DIR_out <- sprintf("%scasestudy2/casestudy2_updfits2_koyck/rolling_fits_ylags_%s_zlags_%s_xlags_%s_hashrateAlmon_%s_btccloseAlmon_%s_boxCox_%s_lamda_%s_step_%s_win_%s/", DIR, ylags,
                               zlags, xlags, hashrate_almonlag, btcclose_almonlag, boxCox, boxCox_lamda, stepsize, window_size)
          } else {
            DIR_out <- sprintf("%scasestudy2/casestudy2_updfits2_koyck/rolling_fits_ylags_%s_zlags_%s_xlags_%s_hashrateAlmon_%s_btccloseAlmon_%s_boxCox_%s_step_%s_win_%s/", DIR, ylags,
                               zlags, xlags, hashrate_almonlag, btcclose_almonlag, boxCox, stepsize, window_size)
          }
          print(DIR_out)
          dir.create(DIR_out)

          window_start_dates <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))

          coef_ivlag1 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          coef_ivlagp <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          coef_ivlag2 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          coef_ivlag3 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          coef_ivlag4 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          coef_ivlag5 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))

          z_theta1 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          z_almonlag1 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          z_almonlag2 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))

          x_theta1 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          x_almon_1 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))

          intercepts <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))

          ######################################################################
          error_ivlag1 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          error_ivlagp <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          error_ivlag2 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          error_ivlag3 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          error_ivlag4 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          error_ivlag5 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))

          error_z_theta1 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          error_z_almonlag1 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          error_z_almonlag2 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))

          error_x_theta1 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          error_x_almonlag1 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))

          error_intercepts <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))

          ######################################################################
          significance_ivlag1 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          significance_ivlagp <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          significance_ivlag2 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          significance_ivlag3 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          significance_ivlag4 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          significance_ivlag5 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))

          significance_z_theta1 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          significance_z_almonlag1 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          significance_z_almonlag2 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))

          significance_x_theta1 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          significance_x_almonlag1 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))

          significance_intercepts <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))

          ######################################################################
          residuals_fits <- matrix(-1000, nrow = length(y) , ncol=(1 + floor((length(y)-window_size)/stepsize)))
          fittedvals <- matrix(-1000, nrow = length(y) , ncol=(1 + floor((length(y)-window_size)/stepsize)))

          y_pvalues <- matrix(-1000, nrow = (ylags+1) , ncol=(1 + floor((length(y)-window_size)/stepsize)))
          pvalues_z_theta1 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          pvalues_z_almonlag1 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          pvalues_z_almonlag2 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))

          pvalues_x_theta1 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
          pvalues_x_almonlag1 <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))

          z_almon_lags_coefs <- matrix(-1000, nrow = (zlags*24) , ncol=(1 + floor((length(y)-window_size)/stepsize)))
          x_almon_lags_coefs <- matrix(-1000, nrow = (xlags*30) , ncol=(1 + floor((length(y)-window_size)/stepsize)))

          ######################################################################
          aics <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))
          mses <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))
          # mses_forecast <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))
          convergence <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))

          y_longmem_hal <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))
          y_longmem_he <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))
          y_longmem_hs <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))
          y_longmem_hrs <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))
          y_longmem_ht <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))

          z_longmem_hal <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))
          x_longmem_hal <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))
          residuals_longmem_hal <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))

          z_longmem_he <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))
          x_longmem_he <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))
          residuals_longmem_he <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))

          z_longmem_hs <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))
          x_longmem_hs <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))
          residuals_longmem_hs <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))

          z_longmem_hrs <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))
          x_longmem_hrs <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))
          residuals_longmem_hrs <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))

          z_longmem_ht <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))
          x_longmem_ht <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))
          residuals_longmem_ht <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))

          k <- 1
          for (i in seq(1, length(y), stepsize) ) {
            # if (as.Date(dates[i]) < as.Date("2018-01-01")){
            #    print(dates[i])
            #    next
            # }
            print(sprintf("******************************  Window start: %s  *********************************************",
                              dates[i]))
            print(i)

            yy <- y[i:(i+window_size-1)]
            iviv <- iv[i:(i+window_size-1)]

            if (length(which(is.na(yy))) > 0){
                break
            }

            xx <- x[i:(i+window_size-1)]

            # print(length(xx))

            stopifnot(dates[i:(i+window_size-1)]==mining_dates[i:(i+window_size-1)])
            if (smoothed){
              zz <- z[(i*24+1):((i+window_size)*24)]
              stopifnot(dates[i:(i+window_size-1)][1]==as.Date(zdates[(i*24+1):((i+window_size)*24)][1]))
              stopifnot(dates[i:(i+window_size-1)][window_size]==as.Date(zdates[(i*24+1):((i+window_size)*24)][window_size*24]))
            } else {
                zz <- z[((i-1)*24+1):((i+window_size-1)*24)]
                stopifnot(dates[i:(i+window_size-1)][1]==as.Date(zdates[((i-1)*24+1):((i+window_size-1)*24)][1]))
                stopifnot(dates[i:(i+window_size-1)][window_size]==as.Date(zdates[((i-1)*24+1):((i+window_size-1)*24)][window_size*24]))
            }

             # print(length(zz))

            if (boxCox){
              if (smoothed){
                y <- btc_sent_entropy[14:length(btc_sent_entropy)]
              } else {
                y <- btc_sent_entropy[8:length(btc_sent_entropy)]
              }
              if (boxCox_lamda != 0){
                y <- (y^boxCox_lamda - 1)/boxCox_lamda
              } else {
                y <- log(y)
              }
              y <- y - mean(y)
              yy <- y[i:(i+window_size-1)]
            }

            converged <- tryCatch({
                          eqr <- midas_r(yy ~ mls(ivivst, 1:1, 1) + mls(ivivp, (ylags+1):(ylags+1), 1) + mls(iviv, 2:ylags, 1) + mls(zz, 1:(zlags*24), 24, nealmon) + mls(xx, 1:(xlags*30), 1, nealmon),
                                  data=list(iviv=iviv, ivivp=iviv, ivivst=iviv, yy=yy, zz=zz, xx=xx), Ofunction = "optim", method="Nelder-Mead",
                                  start=list(zz=rep(-0.1, (btcclose_almonlag+1)), xx=rep(-0.1, (hashrate_almonlag+1))))
                          summary(eqr)
                          alg <- "Nelder-Mead"
                          converged <- eqr$opt$convergence
                          }, error=function(n){
                             return (1)
                          })
            if (converged == 1){
              converged <- tryCatch({
                      eqr <- midas_r(yy ~ mls(ivivst, 1:1, 1) + mls(ivivp, (ylags+1):(ylags+1), 1) + mls(iviv, 2:ylags, 1) + mls(zz, 1:(zlags*24), 24, nealmon) + mls(xx, 1:(xlags*30), 1, nealmon),
                                  data=list(iviv=iviv, ivivp=iviv, ivivst=iviv, yy=yy, zz=zz, xx=xx), Ofunction = "optim", method="BFGS",
                                  start=list(zz=rep(-0.1, (btcclose_almonlag+1)), xx=rep(-0.1, (hashrate_almonlag+1))))
                      summary(eqr)
                      alg <- "BFGS"
                      converged <- eqr$opt$convergence
                      }, error=function(n){
                         return (1)
                      })
            }
            if (converged == 1){
              converged <- tryCatch({
                    eqr <- midas_r(yy ~ mls(ivivst, 1:1, 1) + mls(ivivp, (ylags+1):(ylags+1), 1) + mls(iviv, 2:ylags, 1) + mls(zz, 1:(zlags*24), 24, nealmon) + mls(xx, 1:(xlags*30), 1, nealmon),
                                  data=list(iviv=iviv, ivivp=iviv, ivivst=iviv, yy=yy, zz=zz, xx=xx), Ofunction = "optim", method="BFGS",
                                  start=list(zz=rep(-0.1, (btcclose_almonlag+1)), xx=rep(-0.1, (hashrate_almonlag+1))),
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
            if (converged == 1){
              # move to next window if model did not fit
              next
            }
            summ <- coef(summary(eqr))
            print(summ)
            #
            # # if (length(summ) != 40){
            # #   print("Fit failed...")
            # #   next
            # # }
            # # print(summ)
            # # stopifnot(length(summ) == 40)
            #
            if (length(coef(eqr, midas=FALSE, "iviv"))!=(ylags-1)){
              next
            }
            if (length(coef(eqr, midas=FALSE, "ivivst"))!=1){
              next
            }
            if (length(coef(eqr, midas=FALSE, "ivivp"))!=1){
              next
            }
            if (length(coef(eqr, midas=FALSE, "xx"))!=(hashrate_almonlag+1)){
              next
            }
            if (length(coef(eqr, midas=FALSE, "zz"))!=(btcclose_almonlag+1)){
              next
            }
            if (length(coef(eqr, midas=TRUE, "xx"))!=(xlags*30)){
              next
            }
            if (length(coef(eqr, midas=TRUE, "zz"))!=(zlags*24)){
              next
            }

            window_start_dates[k] <- dates[i]

            mse <- mean((fitted(eqr) - yy[(xlags*30+1):length(yy)])^2)
            aic <- AIC(eqr)

            coef_ivlag1[k] <- coef(eqr, midas=FALSE, "ivivst")
            coef_ivlagp[k] <- coef(eqr, midas=FALSE, "ivivp")
            coef_ivlag2[k] <- coef(eqr, midas=FALSE, "iviv")[1]
            coef_ivlag3[k] <- coef(eqr, midas=FALSE, "iviv")[2]
            coef_ivlag4[k] <- coef(eqr, midas=FALSE, "iviv")[3]
            coef_ivlag5[k] <- coef(eqr, midas=FALSE, "iviv")[4]

            y_pvalues[, k] <- summ[2:(ylags+2), 4]

            z_theta1[k] <- coef(eqr, midas=FALSE, "zz")[1]
            z_almonlag1[k] <- coef(eqr, midas=FALSE, "zz")[2]

            pvalues_z_theta1[k] <- summ[1+ylags+1+1, 4] # MIDAS false
            pvalues_z_almonlag1[k] <- summ[1+ylags+1+1+1, 4]
            if (btcclose_almonlag==2){
              z_almonlag2[k] <- coef(eqr, midas=FALSE, "zz")[3]
              pvalues_z_almonlag2[k] <- summ[1+ylags+1+1+1+1, 4]
            }

            x_theta1[k] <- coef(eqr, midas=FALSE, "xx")[1]
            x_almon_1[k] <- coef(eqr, midas=FALSE, "xx")[2]

            pvalues_x_theta1[k] <- summ[(1+ylags+btcclose_almonlag+1+1+1), 4] # MIDAS false
            pvalues_x_almonlag1[k] <- summ[(1+ylags+btcclose_almonlag+1+1+1+1), 4]

            intercepts[k] <- coef(eqr, midas=FALSE)[1]

            z_almon_lags_coefs[, k] <- coef(eqr, midas=TRUE, "zz")
            x_almon_lags_coefs[, k] <- coef(eqr, midas=TRUE, "xx")

            ######################################################################
            error_ivlag1[k] <- summ[, 2][2]
            error_ivlagp[k] <- summ[, 2][3]
            error_ivlag2[k] <- summ[, 2][4]
            error_ivlag3[k] <- summ[, 2][5]
            error_ivlag4[k] <- summ[, 2][6]
            error_ivlag5[k] <- summ[, 2][7]

            error_z_theta1[k] <- summ[, 2][8]
            error_z_almonlag1[k] <- summ[, 2][9]
            if (btcclose_almonlag==2){
              error_z_almonlag2[k] <- summ[, 2][10]
            }

            if (btcclose_almonlag==2){
              error_x_theta1[k] <- summ[, 2][11]
              error_x_almonlag1[k] <- summ[, 2][12]
            } else {
              error_x_theta1[k] <- summ[, 2][10]
              error_x_almonlag1[k] <- summ[, 2][11]
            }

            error_intercepts[k] <- summ[, 2][1]

            ######################################################################
            significance_ivlag1[k] <- star_significance(summ[, 4][2])
            significance_ivlagp[k] <- star_significance(summ[, 4][3])
            significance_ivlag2[k] <- star_significance(summ[, 4][4])
            significance_ivlag3[k] <- star_significance(summ[, 4][5])
            significance_ivlag4[k] <- star_significance(summ[, 4][6])
            significance_ivlag5[k] <- star_significance(summ[, 4][7])

            significance_z_theta1[k] <- star_significance(summ[, 4][8])
            significance_z_almonlag1[k] <- star_significance(summ[, 4][9])

             if (btcclose_almonlag==2){
              significance_z_almonlag2[k] <- star_significance(summ[, 4][10])
             }

             if (btcclose_almonlag==2){
               significance_x_theta1[k] <- star_significance(summ[, 4][11])
               significance_x_almonlag1[k] <- star_significance(summ[, 4][12])
             } else {
               significance_x_theta1[k] <- star_significance(summ[, 4][10])
               significance_x_almonlag1[k] <- star_significance(summ[, 4][11])
             }

            significance_intercepts[k] <- star_significance(summ[, 4][1])

            ######################################################################
            aics[k] <- aic
            mses[k] <- mse
            convergence[k] <- converged

            # get corrected empirical R/S Hurst exponent
            yy_rs <- hurstexp(yy, d=50, display=TRUE)
            y_longmem_hal[k] <- yy_rs$Hal
            y_longmem_hs[k] <- yy_rs$Hs
            y_longmem_hrs[k] <- yy_rs$Hrs
            y_longmem_he[k] <- yy_rs$He
            y_longmem_ht[k] <- yy_rs$Ht
            zz_rs <- hurstexp(zz, d=50, display=TRUE)
            z_longmem_hal[k] <- zz_rs$Hal
            z_longmem_hs[k] <- zz_rs$Hs
            z_longmem_hrs[k] <- zz_rs$Hrs
            z_longmem_he[k] <- zz_rs$He
            z_longmem_ht[k] <- zz_rs$Ht

            xx_rs <- hurstexp(xx, d=50, display=TRUE)
            x_longmem_hal[k] <- xx_rs$Hal
            x_longmem_hs[k] <- xx_rs$Hs
            x_longmem_hrs[k] <- xx_rs$Hrs
            x_longmem_he[k] <- xx_rs$He
            x_longmem_ht[k] <- xx_rs$Ht

            res_rs <- hurstexp(residuals(eqr), d=50, display=TRUE)
            residuals_longmem_hal[k] <- res_rs$Hal
            residuals_longmem_hs[k] <- res_rs$Hs
            residuals_longmem_hrs[k] <- res_rs$Hrs
            residuals_longmem_he[k] <- res_rs$He
            residuals_longmem_ht[k] <- res_rs$Ht

            # Forecasts
            fcast <- tryCatch({
                zzz <- z[((i+window_size)*24+1):((i+stepsize+window_size)*24)]
                xxx <- x[(i+window_size+1):(i+stepsize+window_size)]
                forecast_days <- dates[(i+window_size+1):(i+stepsize+window_size)]
                win_forecast <- forecast(eqr, newdata=list(zz=zzz, xx=xxx),
                                         se=TRUE, level=c(95), method="dynamic", show_progress=TRUE)

                mses_forecast[k] <- mean((fitted(win_forecast) - y[(i+window_size+1):(i+stepsize+window_size)])^2)

                forecast_mean <- rep(NA, stepsize)
                forecast_upperCI <- rep(NA, stepsize)
                forecast_lowerCI <- rep(NA, stepsize)
                forecast_mean[1:stepsize] <- win_forecast$mean
                forecast_upperCI[1:stepsize] <- win_forecast$upper
                forecast_lowerCI[1:stepsize] <- win_forecast$lower
                data_window <- data.frame(forecast1month=forecast_mean,
                                          forecast1month_upperCI=forecast_upperCI,
                                          forecast1month_lowerCI=forecast_lowerCI,
                                          forecast_days=forecast_days)
                write.csv(data_window, sprintf("%srollingfit_dateStart_%s.csv",DIR_out, dates[i]), row.names = FALSE)
            }, error=function(n){
                       return (1)
                    })
            if (!boxCox){
              jpeg(sprintf("%srolling_midas_fit_%s_Residuals_norm_BoxCox_%s.jpeg",DIR_out, dates[i], boxCox))
              plot(residuals(eqr), type="l")
              dev.off()
              jpeg(sprintf("%srolling_midas_fit_%s_QQ_norm_BoxCox_%s.jpeg",DIR_out, dates[i], boxCox))
              qqPlot(residuals(eqr), distribution="norm")
              dev.off()
            } else{
              jpeg(sprintf("%srolling_midas_fit_%s_Residuals_norm_BoxCox_%s_lamda_%s.jpeg",DIR_out, dates[i], boxCox, boxCox_lamda))
              plot(residuals(eqr), type="l")
              dev.off()
              jpeg(sprintf("%srolling_midas_fit_%s_QQ_norm_BoxCox_%s_lamda_%s.jpeg",DIR_out, dates[i], boxCox, boxCox_lamda))
              qqPlot(residuals(eqr), distribution="norm")
              dev.off()
            }
            resid <- residuals(eqr)
            residuals_fits[(1:length(resid)), k] <- resid
            ftvals <- fitted(eqr)
            fittedvals[(1:length(ftvals)), k] <- ftvals

            k <- k + 1

          }



          data_global_ylongmem <- data.frame(y_longmem_hal=y_longmem_hal,
                                             y_longmem_ht=y_longmem_ht,
                                             y_longmem_he=y_longmem_he,
                                             y_longmem_hrs=y_longmem_hrs,
                                             y_longmem_hs=y_longmem_hs)
          write.csv(data_global_ylongmem, sprintf("%srollingfit_window_%s_BoxCox_%s_BoxCox_Lamda_%s_ylongmem.csv",DIR_out,
                                         window_size, boxCox, boxCox_lamda), row.names = FALSE)

          data_global <- data.frame(window_start_dates=window_start_dates,
                                    coef_ivlag1=coef_ivlag1,
                                    coef_ivlagp=coef_ivlagp,
                                    coef_ivlag2=coef_ivlag2,
                                    coef_ivlag3=coef_ivlag3,
                                    coef_ivlag4=coef_ivlag4,
                                    coef_ivlag5=coef_ivlag5,
                                    z_theta1=z_theta1,
                                    z_almonlag1=z_almonlag1,
                                    z_almonlag2=z_almonlag2,
                                    x_theta1=x_theta1,
                                    x_almon_1=x_almon_1,
                                    intercepts=intercepts,
                                    error_ivlag1=error_ivlag1,
                                    error_ivlagp=error_ivlagp,
                                    error_ivlag2=error_ivlag2,
                                    error_ivlag3=error_ivlag3,
                                    error_ivlag4=error_ivlag4,
                                    error_ivlag5=error_ivlag5,
                                    error_z_theta1=error_z_theta1,
                                    error_z_almonlag1=error_z_almonlag1,
                                    error_z_almonlag2=error_z_almonlag2,
                                    error_x_theta1=error_x_theta1,
                                    error_x_almonlag1=error_x_almonlag1,
                                    error_intercepts=error_intercepts,
                                    significance_ivlag1=significance_ivlag1,
                                    significance_ivlagp=significance_ivlagp,
                                    significance_ivlag2=significance_ivlag2,
                                    significance_ivlag3=significance_ivlag3,
                                    significance_ivlag4=significance_ivlag4,
                                    significance_ivlag5=significance_ivlag5,
                                    significance_z_theta1=significance_z_theta1,
                                    significance_z_almonlag1=significance_z_almonlag1,
                                    significance_z_almonlag2=significance_z_almonlag2,
                                    significance_x_theta1=significance_x_theta1,
                                    significance_x_almonlag1=significance_x_almonlag1,
                                    significance_intercepts=significance_intercepts,
                                    pvalues_z_almonlag1=pvalues_z_almonlag1,
                                    pvalues_z_almonlag2=pvalues_z_almonlag2,
                                    pvalues_x_almonlag1=pvalues_x_almonlag1,
                                    pvalues_z_theta1=pvalues_z_theta1,
                                    pvalues_x_theta1=pvalues_x_theta1,
                                    aics=aics,
                                    mses=mses,
                                    convergence=convergence,
                                    z_longmem_hal=z_longmem_hal,
                                    x_longmem_hal=x_longmem_hal,
                                    residuals_longmem_hal=residuals_longmem_hal,
                                    z_longmem_ht=z_longmem_ht,
                                    x_longmem_ht=x_longmem_ht,
                                    residuals_longmem_ht=residuals_longmem_ht,
                                    z_longmem_he=z_longmem_he,
                                    x_longmem_he=x_longmem_he,
                                    residuals_longmem_he=residuals_longmem_he,
                                    z_longmem_hrs=z_longmem_hrs,
                                    x_longmem_hrs=x_longmem_hrs,
                                    residuals_longmem_hrs=residuals_longmem_hrs,
                                    z_longmem_hs=z_longmem_hs,
                                    x_longmem_hs=x_longmem_hs,
                                    residuals_longmem_hs=residuals_longmem_hs)
          write.csv(data_global, sprintf("%srollingfit_window_%s_BoxCox_%s_BoxCox_Lamda_%s.csv",DIR_out,
                                         window_size, boxCox, boxCox_lamda), row.names = FALSE)
          y_res <- data.frame(residuals=residuals_fits)
          write.csv(y_res, sprintf("%srollingfits_residuals_startdate_%s.csv",DIR_out, dates[i]), row.names = FALSE)
          y_fits <- data.frame(fitted=fittedvals)
          write.csv(y_fits, sprintf("%srollingfits_fitted.csv", DIR_out), row.names = FALSE)
          y_lagspvalues <- data.frame(y_pvalues=y_pvalues)
          write.csv(y_lagspvalues, sprintf("%srollingfit_window_%s_BoxCox_%s_BoxCox_Lamda_%s_Y_pvalues.csv",DIR_out,
                                         window_size, boxCox, boxCox_lamda), row.names = FALSE)
          z_almonlags <- data.frame(z_almon_lags_coefs=z_almon_lags_coefs)
          write.csv(z_almonlags, sprintf("%srollingfit_window_%s_BoxCox_%s_BoxCox_Lamda_%s_MIDAS_true_Z_almon_coefs.csv",DIR_out,
                                         window_size, boxCox, boxCox_lamda), row.names = FALSE)
          x_almonlags <- data.frame(x_almon_lags_coefs=x_almon_lags_coefs)
          write.csv(x_almonlags, sprintf("%srollingfit_window_%s_BoxCox_%s_BoxCox_Lamda_%s_MIDAS_true_X_almon_coefs.csv",DIR_out,
                                         window_size, boxCox, boxCox_lamda), row.names = FALSE)
    }
}