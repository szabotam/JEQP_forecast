{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# JEQP ETF 30 napos árfolyam előrejelzés (Prophet)\n",
    "\n",
    "Ez a notebook a JEQP ETF napi árfolyamát elemzi, és a Facebook Prophet segítségével 30 napos előrejelzést készít."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install yfinance prophet matplotlib --quiet"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import yfinance as yf\n",
    "import pandas as pd\n",
    "from prophet import Prophet\n",
    "import matplotlib.pyplot as plt\n",
    "from datetime import datetime\n",
    "\n",
    "# JEQP adatlekérés Yahoo Finance-ről\n",
    "ticker = 'JEQP'\n",
    "data = yf.download(ticker, start='2020-01-01')\n",
    "\n",
    "# Prophet formátumra alakítás\n",
    "df = data[['Close']].reset_index()\n",
    "df.columns = ['ds', 'y']\n",
    "\n",
    "# Prophet modell\n",
    "model = Prophet(daily_seasonality=True)\n",
    "model.fit(df)\n",
    "\n",
    "# 30 napos előrejelzés\n",
    "future = model.make_future_dataframe(periods=30)\n",
    "forecast = model.predict(future)\n",
    "\n",
    "# Grafikon\n",
    "fig1 = model.plot(forecast)\n",
    "plt.title('JEQP ETF - 30 napos előrejelzés')\n",
    "plt.show()\n",
    "\n",
    "# Előrejelzés táblázat mentése\n",
    "forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('JEQP_forecast.csv', index=False)\n",
    "print('Előrejelzés mentve: JEQP_forecast.csv')"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "name": "JEQP_forecast.ipynb"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
