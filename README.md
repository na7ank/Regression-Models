# Regressão Linear Símples :blue_book:
## Implementação do Método de Mínimos Quadrados e Métricas para Modelo de Regressão Linear com Python
O método dos mínimos quadrados é o mesmo que reta de regressão linear símples ou reta de ajuste.
Interessante estudar esse modelo porque temos aplicações em Machine Learning 🦾.

![Imagem 1](/imgs/exemplo_gif.gif)



## Método dos Mínimos Quadrados ✔️

O método dos mínimos quadrados é utilizado para ajustar um modelo de regressão linear aos dados observados, minimizando a soma dos quadrados dos resíduos (as diferenças entre os valores observados e os valores preditos).
Para uma regressão linear simples com uma variável independente, a fórmula dos coeficientes é:

**Coeficiente Angular $A$**

$$ 
A = {\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y}) } / { \sum_{i=1}^{n} (x_i - \bar{x})^2 }
$$

**Coeficiente Linear $B$**

$$ B = \bar{y} - A \bar{x} $$

onde:
- **$x_i$**  são os valores da variável independente,
- **$y_i$** são os valores observados da variável dependente,
- **$\bar{x}$** é a média dos valores da variável independente $$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$,
- **$\bar{y}$** é a média dos valores da variável dependente $$\bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i$$,
- **$n$** é o número de observações.

### Equação da Regressão Linear
A equação da reta de regressão linear ajustada é dada por:

$$ \hat{y} = B + A x $$

onde:
- **$\hat{y}$** é o valor predito da variável dependente,
- **$A$** é o coeficiente angular,
- **$B$** é o coeficiente linear,
- **$x$** é o valor da variável independente.

Usaremos essas fórmulas para calcular os coeficientes que melhor se ajustam aos dados, minimizando a soma dos quadrados das diferenças entre os valores observados e os valores preditos 🤓.

## Métricas Performance ✔️

### Raiz do Erro Quadrático Médio (RMSE)

A Raiz do Erro Quadrático Médio (RMSE) é uma métrica que mede a média dos erros ao quadrado entre os valores observados e os valores preditos (MSE). Ela tem a mesma unidade dos valores observados, o que facilita a interpretação.

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

onde:
- **$y_i$** são os valores observados,
- **$\hat{y}_i$** são os valores preditos pelo modelo, para cada **$y_i$**,
- **$n$** é o número de observações.

### Coeficiente de Determinação R²

O Coeficiente de Determinação R² é uma métrica que mede a proporção da variabilidade dos dados que é explicada pelo modelo de regressão. Ele varia entre 0 e 1, onde 1 indica um ajuste perfeito e 0 indica que o modelo não explica a variabilidade dos dados.

$$ 
R² = 1 - { \sum_{i=1}^{n} (y_i - \hat{y_i})} / { \sum_{i=1}^{n} (y_i - \bar{y})^2 }
$$


onde:
- **$y_i$** são os valores observados,
- **$\hat{y}_i$** são os valores preditos pelo modelo,
- **$\bar{y}$** é a média dos valores observados $$\bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i$$,
- **$n$** é o número de observações.

# Regressão Logística Binária :closed_book:

### A fórmula geral para o modelo de regressão logística binária é:

$$ P(Y = 1 \mid \mathbf{X}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + \beta_2X_2 + \cdots + \beta_nX_n)}} $$

onde:
- $P(Y = 1 \mid \mathbf{X})$ é a probabilidade de o evento ocorrer (Y = 1) dado o vetor de características $ \mathbf{X} $.
- $ \beta_0 $ é o intercepto do modelo.
- $ \beta_1 $, $ \beta_2 $, $ \ldots $, $ \beta_n $ são os coeficientes das variáveis independentes $ X_1, X_2, \ldots , X_n $ (Vetor de características).
- $e$ é a base do logaritmo natural (aproximadamente 2.71828).

### Forma Logit

Podemos reescrever a equação na forma logit, que é a função log-odds (log das chances):

$$ \text{logit}(P(Y = 1 \mid \mathbf{X})) = \ln\left(\frac{P(Y = 1 \mid \mathbf{X})}{1 - P(Y = 1 \mid \mathbf{X})}\right) = \beta_0 + \beta_1X_1 + \beta_2X_2 + \cdots + \beta_nX_n $$

Nesta forma, a regressão logística se parece muito com a regressão linear, mas está modelando o log das chances do evento \(Y = 1\).

### Interpretação dos Coeficientes

- **$\beta_0$ (intercepto):** É o log-odds de $Y = 1$ quando todas as variáveis $X_i$ são $0$.
- **$\beta_i$ (coeficientes das variáveis):** Representa a mudança no log-odds de $Y = 1$ para uma unidade de mudança na variável $X_i$, mantendo todas as outras variáveis constantes.


