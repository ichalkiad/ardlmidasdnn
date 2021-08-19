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
gammas <- c(0.01, 0.05, 0.1)
gammas <- append(gammas, seq(0.15, 0.9, by=0.05), 3)

for (config in configurations) {
  for (gamma in gammas) {
      print(gamma)
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
      # c(720, 900, 1080, 1260)

      for (window_size in winsizes){

            if (boxCox) {
              DIR_out <- sprintf("%scasestudy2/casestudy2_updfits2_koyck_forecast/rolling_fits_ylags_%s_zlags_%s_xlags_%s_hashrateAlmon_%s_btccloseAlmon_%s_boxCox_%s_lamda_%s_step_%s_win_%s_gamma_%s/", DIR, ylags,
                                 zlags, xlags, hashrate_almonlag, btcclose_almonlag, boxCox, boxCox_lamda, stepsize, window_size, gamma)
            } else {
              DIR_out <- sprintf("%scasestudy2/casestudy2_updfits2_koyck_forecast/rolling_fits_ylags_%s_zlags_%s_xlags_%s_hashrateAlmon_%s_btccloseAlmon_%s_boxCox_%s_step_%s_win_%s_gamma_%s/", DIR, ylags,
                                 zlags, xlags, hashrate_almonlag, btcclose_almonlag, boxCox, stepsize, window_size, gamma)
            }
            print(DIR_out)
            dir.create(DIR_out)

            window_start_dates <- rep(-1000, 1 + floor((length(y)-window_size)/stepsize))
            mses_forecast <- rep(100000, 1 + floor((length(y)-window_size)/stepsize))

            k <- 1
            for (i in seq(1, length(y), stepsize) ) {

              print(sprintf("******************************  Window start: %s  *********************************************",
                                dates[i]))
              print(i)

              # response and instrumental variable in current window
              yy <- y[i:(i+window_size-1)]
              iviv <- iv[i:(i+window_size-1)]

              if (length(which(is.na(yy))) > 0){
                  break
              }

              # hashrate
              xx <- x[i:(i+window_size-1)]

              stopifnot(dates[i:(i+window_size-1)]==mining_dates[i:(i+window_size-1)])
              # BTC close price
              if (smoothed){
                zz <- z[(i*24+1):((i+window_size)*24)]
                stopifnot(dates[i:(i+window_size-1)][1]==as.Date(zdates[(i*24+1):((i+window_size)*24)][1]))
                stopifnot(dates[i:(i+window_size-1)][window_size]==as.Date(zdates[(i*24+1):((i+window_size)*24)][window_size*24]))
              } else {
                  zz <- z[((i-1)*24+1):((i+window_size-1)*24)]
                  stopifnot(dates[i:(i+window_size-1)][1]==as.Date(zdates[((i-1)*24+1):((i+window_size-1)*24)][1]))
                  stopifnot(dates[i:(i+window_size-1)][window_size]==as.Date(zdates[((i-1)*24+1):((i+window_size-1)*24)][window_size*24]))
              }

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
                          eqr <- midas_r(yy ~  mls(yy, 1:ylags, 1) + mls(zz, 1:(zlags*24), 24, nealmon) + mls(xx, 1:(xlags*30), 1, nealmon),
                                  data=list(yy=yy, zz=zz, xx=xx), Ofunction = "optim", method="Nelder-Mead",
                                  start=list(zz=rep(-0.1, (btcclose_almonlag+1)), xx=rep(-0.1, (hashrate_almonlag+1))))
                          summary(eqr)
                          alg <- "Nelder-Mead"
                          converged <- eqr$opt$convergence
                          }, error=function(n){
                             return (1)
                          })
              if (converged == 1){
                converged <- tryCatch({
                        eqr <- midas_r(yy ~  mls(yy, 1:ylags, 1) + mls(zz, 1:(zlags*24), 24, nealmon) + mls(xx, 1:(xlags*30), 1, nealmon),
                                    data=list(yy=yy, zz=zz, xx=xx), Ofunction = "optim", method="BFGS",
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
                      eqr <- midas_r(yy ~  mls(yy, 1:ylags, 1) + mls(zz, 1:(zlags*24), 24, nealmon) + mls(xx, 1:(xlags*30), 1, nealmon),
                                    data=list(yy=yy, zz=zz, xx=xx), Ofunction = "optim", method="BFGS",
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

              if (length(coef(eqr, midas=FALSE, "yy"))!=ylags){
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

              # errors for current fit and adjsusted with Koyck transform
              errors <- rnorm(2*length(yy), mean=0, sd=summary(eqr)$sigma)
              innov <- errors[2:length(errors)] - gamma*errors[1:(length(errors)-1)]
              err_idx <- 0
              window_start_dates[k] <- dates[i]

              # z almon weights
              if (btcclose_almonlag == 1) {
                theta_z <- c(coef(eqr, midas=FALSE, "zz")[1], coef(eqr, midas=FALSE, "zz")[2])
              } else {
                theta_z <- c(coef(eqr, midas=FALSE, "zz")[1], coef(eqr, midas=FALSE, "zz")[2], coef(eqr, midas=FALSE, "zz")[3])
              }
              # MIDAS weights for BTC close price
              z_params <- nealmon(theta_z, (zlags*24))

              # x almon weights
              theta_x <- c(coef(eqr, midas=FALSE, "xx")[1], coef(eqr, midas=FALSE, "xx")[2])
              # MIDAS weights for hash rate
              x_params <- nealmon(theta_x, (xlags*30))

              # load parameters of fitted model and adjust with Koyck transform to get MIDAS-Koyck parameters
              intercept_prime <- (1-gamma)*summ[1, 1]
              phi_prime <- summ[2, 1] + gamma
              phi_2prime <- -gamma*summ[(1+ylags), 1]
              varphi2 <- summ[3, 1] - gamma*summ[2, 1]
              varphi3 <- summ[4, 1] - gamma*summ[3, 1]
              varphi4 <- summ[5, 1] - gamma*summ[4, 1]
              varphi5 <- summ[6, 1] - gamma*summ[5, 1]

              y_forecast <- rep(NA, stepsize)
              forecast_days <- rep(NA, stepsize)
              # forecast next 'stepsize' days one at a time, every time using the forecasted value as the previous response value
              for (jj in seq(1:stepsize)){

                   if (jj == 1){
                     ylag1 <- phi_prime * iviv[length(iviv)]
                   } else{
                     ylag1 <- phi_prime * y_forecast[jj-1]
                   }
                   if (jj <= ylags){
                     ylagp <- phi_2prime * iviv[length(iviv)-1-ylags+jj]
                   } else{
                     ylagp <- phi_2prime * y_forecast[jj-ylags]
                   }
                   if (jj <= 2){
                     ylag2 <- varphi2 * iviv[length(iviv)-2+jj]
                   } else{
                     ylag2 <- varphi2 * y_forecast[jj-2]
                   }
                   if (jj <= 3){
                     ylag3 <- varphi3 * iviv[length(iviv)-3+jj]
                   } else{
                     ylag3 <- varphi3 * y_forecast[jj-3]
                   }
                   if (jj <= 4){
                     ylag4 <- varphi4 * iviv[length(iviv)-4+jj]
                   } else{
                     ylag4 <- varphi4 * y_forecast[jj-4]
                   }
                   if (jj <= 5){
                     ylag5 <- varphi5 * iviv[length(iviv)-5+jj]
                   } else{
                     ylag5 <- varphi5 * y_forecast[jj-5]
                   }

                   zzz <- tail(z[(i*24+1):((i+window_size+jj)*24)], (zlags*24))
                   xxx <- tail(x[(i+1):(i+window_size+jj)], (xlags*30))
                   forecast_days[jj] <- dates[i+window_size+jj]

                   y_forecast[jj] <- intercept_prime + ylag1 + ylagp + ylag2 + ylag3 + ylag4 + ylag5 +
                         zzz%*%z_params + xxx%*%x_params + innov[err_idx+jj]

              }
              err_idx <- err_idx + stepsize

              mses_forecast[k] <- mean((y_forecast - y[(i+window_size+1):(i+window_size+stepsize)])^2)

              data_window <- data.frame(forecast1month=y_forecast)
              write.csv(data_window, sprintf("%srollingfit_dateStart_%s.csv",DIR_out, dates[i]), row.names = FALSE)


              k <- k + 1

            }
              mse_out <- data.frame(dates=window_start_dates,
                                    mses_forecast=mses_forecast)
              write.csv(mse_out, sprintf("%srollingfit_MIDAS_Koyck_forecasts_gamma_%s.csv",DIR_out, gamma), row.names = FALSE)
      }
  }
}