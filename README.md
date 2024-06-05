## Implementação do Método de Mínimos Quadrados e Métricas para Modelo de Regressão Linear com Python
O método dos mínimos quadrados é o mesmo que reta de regressão linear símples ou reta de ajuste.
Interessante estudar esse modelo porque temos aplicações em Machine Learning 🦾.

![Amostras](/imgs/output.png)


## Método dos Mínimos Quadrados ✔️

O método dos mínimos quadrados é utilizado para ajustar um modelo de regressão linear aos dados observados, minimizando a soma dos quadrados dos resíduos (as diferenças entre os valores observados e os valores preditos).
Para uma regressão linear simples com uma variável independente, a fórmula dos coeficientes é:

**Coeficiente Angular $A$**

$$ 
A = \frac { \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y}) } { \sum_{i=1}^{n} (x_i - \bar{x})^2 }
$$

**Coeficiente Linear $B$**

$$ B = \bar{y} - A \bar{x} $$

onde:
- **$x_i$**  são os valores da variável independente,
- **$y_i$** são os valores observados da variável dependente,
- **$\bar{x}$** é a média dos valores da variável independente,
- **$\bar{y}$** é a média dos valores da variável dependente,
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
R^2 = 1 - \frac {\sum_{i=1}^{n} (y_i - \hat{y}_i)^2} {\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

onde:
- **$y_i$** são os valores observados,
- **$\hat{y}_i$** são os valores preditos pelo modelo,
- **$\bar{y}$** é a média dos valores observados,
- **$n$** é o número de observações.


