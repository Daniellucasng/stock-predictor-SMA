# SMA SIMPLE MOVING AVERAGE
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
# MACHINE LEARNING
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Definir el símbolo de la acción
ticket_symbol = 'MSFT'

# Fechas
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=3650)).strftime('%Y-%m-%d')

# Descargar datos
data = yf.download(ticket_symbol, start=start_date, end=end_date)

# Parámetros de medias móviles
long_sma = 40
short_sma = 10

# Calcular SMA larga
sma_long_colname = f'SMA_{long_sma}'
data[sma_long_colname] = data['Close'].rolling(window=long_sma).mean()
delta_sma_long_colname = f'd_{sma_long_colname}'
data[delta_sma_long_colname] = ((data[sma_long_colname] - data[sma_long_colname].shift(1)) / data[sma_long_colname].shift(1)) * 100

# Calcular SMA corta
sma_short_colname = f'SMA_{short_sma}'
data[sma_short_colname] = data['Close'].rolling(window=short_sma).mean()
delta_sma_short_colname = f'd_{sma_short_colname}'
data[delta_sma_short_colname] = ((data[sma_short_colname] - data[sma_short_colname].shift(1)) / data[sma_short_colname].shift(1)) * 100

# Calcular RSI
delta = data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Calcular MACD
ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema_12 - ema_26
data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Limpiar nulos
data = data.dropna()

# Recomendaciones
data['Recomendacion'] = 'ESPERAR'
data.loc[(data[delta_sma_long_colname] > 0) & (data[delta_sma_short_colname] > 0), 'Recomendacion'] = 'COMPRAR'
data.loc[(data[delta_sma_long_colname] < 0) & (data[delta_sma_short_colname] < 0), 'Recomendacion'] = 'VENDER'

# Exportar a Excel
excel_file = f'DATA/{ticket_symbol}_historico_precios.xlsx'
data.to_excel(excel_file)

# Filtrar últimos 365 días
data = data.tail(365)

# Graficar
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index, data[sma_short_colname], label=sma_short_colname, linestyle='-', linewidth=1, color='blue')
ax.plot(data.index, data[sma_long_colname], label=sma_long_colname, linestyle='-', linewidth=2, color='yellow')
ax.plot(data.index, data['Close'], label='Close', linestyle='-', linewidth=1, color='gray')

for index, row in data.iterrows():
    recomendacion = row['Recomendacion']
    if isinstance(recomendacion, pd.Series):
        recomendacion = recomendacion.item()

    color = 'gray'
    if recomendacion == 'COMPRAR':
        color = 'green'
    elif recomendacion == 'VENDER':
        color = 'red'

    if color != 'gray':
        ax.plot(index, row['Close'], marker='o', markersize=4, color=color)


ax.set_title(f'Grafico de SMA_{short_sma} y SMA_{long_sma} y close para {ticket_symbol}')
ax.set_xlabel('Fecha')
ax.set_ylabel('Precio')
ax.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()


# Preparar datos
features = ['Close', sma_short_colname, sma_long_colname, delta_sma_short_colname, delta_sma_long_colname, 'Volume', 'RSI', 'MACD', 'MACD_signal']
X = data[features]
y = data['Recomendacion']

# Codificar etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Entrenar modelo
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicción futura
future_index = data.index[-1] + pd.DateOffset(days=30)
future_row = data.iloc[-1][features]
future_pred = model.predict([future_row])[0]
future_label = label_encoder.inverse_transform([future_pred])[0]

# Simular cambio de precio según recomendación
last_index = data.index[-1]
last_close = data['Close'].iloc[-1]
if future_label == 'COMPRAR':
    future_close = last_close * 1.05
elif future_label == 'VENDER':
    future_close = last_close * 0.95
else:
    future_close = last_close

# Mostrar en gráfica
color_pred = {'COMPRAR': 'green', 'VENDER': 'red', 'ESPERAR': 'gray'}[future_label]
ax.plot(future_index, future_close, marker='x', markersize=8, color=color_pred, label=f'Predicción ({future_label})')

# Agregar texto de recomendación en el punto de predicción
ax.annotate(
    future_label,
    xy=(future_index, future_close),
    xytext=(future_index, future_close * 1.02),  # Ajusta la posición vertical del texto
    color=color_pred,
    fontsize=10,
    fontweight='bold',
    ha='center',
    arrowprops=dict(arrowstyle='->', color=color_pred)
)

ax.plot([last_index, future_index], [last_close, future_close], linestyle='--', color=color_pred, linewidth=1.5, label='Proyección 180 días')

# Mostrar texto en consola
print(f"\n🔮 Predicción para {future_index.date()}: {future_label}")

plt.show()
