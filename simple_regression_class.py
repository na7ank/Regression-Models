import math
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    """
    Recebe um numpy array com as coordenadas das observações [(x1, y1), (x2, y2), ...].
    Queremos criar um modelo de regressão linear e algumas métricas para prever outros valores.
    """
    def __init__(self, coordenadas=None):
        # Pontos Originais ou exemplo
        if coordenadas is not None:
            self.coordenadas_iniciais = coordenadas
        else:
            sinal = np.random.uniform(-1, 1)
            self.coordenadas_iniciais = np.array([[i, sinal*i + np.random.normal(0,10)] for i in range(200)])

        self.numero_pontos = len(self.coordenadas_iniciais)

        # Regressão
        self.coeficiente_angular, self.coeficiente_linear = self.metodo_minimos_quadrados()
        self.pontos_regressao = self.regressao()

        # Metricas
        self.rmse, self.r2 = self.model_metrics()
        
    def metodo_minimos_quadrados(self):
        """
        Recebe uma lista com os pontos (xi, yi)
        representando pontos de valores que queremos ajustar a reta de minimos quadrados
        Retorna (A, B) Onde A e o coeficiente angular e B o linear
        
        n: 1,2,3, ... , n
        S: Somatorio
        
        A = [ n.S(xi.yi) - S(xi)S(yi) ] / [ n.S(xi)^2 - (S(xi))^2 ]
        A_p1 = S(xi.yi)
        A_p2 = S(xi)
        A_p3 = S(yi)
        A_p4 = S(xi)^2
        
        B = [ S(yi) - A.S(xi) ] / n
        B_p1 = S(yi)
        B_p2 = S(xi)
        """
        A_p1 = 0 
        A_p2 = 0
        A_p3 = 0
        A_p4 = 0
        for ponto in self.coordenadas_iniciais:
            xi = ponto[0]
            yi = ponto[1]
            
            A_p1 += xi*yi
            A_p2 += xi
            A_p3 += yi
            A_p4 += (xi*xi)

        A = (self.numero_pontos*A_p1 - A_p2*A_p3) / (self.numero_pontos*A_p4 - A_p2*A_p2)
        B = (A_p3 - A*A_p2) / self.numero_pontos
        
        return (A, B)
    
    def regressao(self):
        """
        Retorna uma lista com os pontos (xi, y^i), 
        Onde y^i é cada valor ajustado para a reta dos mínimos quadrados com coeficientes (A, B)
        """
        pontos_novos = np.array([[ponto[0], self.coeficiente_angular*ponto[0] + self.coeficiente_angular] for ponto in self.coordenadas_iniciais])

        return pontos_novos
    
    def previsao(self, x):
        """
        Retorna um valor para a previsão de quando x
        """
        A = self.coeficiente_angular
        B = self.coeficiente_linear
        y_previsto = A*x + B
        return y_previsto

    def model_metrics(self):
        """
        Cálcula o erro quadrático médio (MSE) e sua raíz (RMSE) para servir de parâmetro de avaliação para o modelo de regressão.
        Calcula o coeficiente de determinação R².
        """
        predict = self.pontos_regressao[:, 1]
        original = self.coordenadas_iniciais[:, 1]
        mean_original = original.mean()

        # Sums
        sum_of_squared_error = ((predict - original)**2).sum()
        sum_of_squared_origin = ((original - mean_original)**2).sum()
        
        # RMSE: Root Mean Squared Error
        mean_squared_error = sum_of_squared_error / self.numero_pontos
        root_mean_squared_error = math.sqrt(mean_squared_error)

        # Coeficiente de Determinação
        r2 = 1 - (sum_of_squared_error / sum_of_squared_origin)
    
        return (root_mean_squared_error, r2)
    
    def grafico(self):
        """
        Mostra um gráfico dos pontos originais e ajustes com informações das métricas
        """
        # Coeficientes
        A = self.coeficiente_angular
        B = self.coeficiente_linear
        # Métricas
        r2 = self.r2
        rmse = self.rmse
        # Plotando Pontos Originais
        x = self.coordenadas_iniciais[:, 0]
        y = self.coordenadas_iniciais[:, 1]
        plt.title('Mínimos Quadrados - Regressão Linear')
        plt.scatter(x, y, color='#0CFA93', alpha=0.6, s=10, label='Observações')
        # Plotando Reta de Regressão
        y_ajuste = [ponto[1] for ponto in self.pontos_regressao]
        plt.plot(x, y_ajuste, label='Ajuste', color='blue', linewidth=0.8)
        # Textos / Legenda
        if B > 0: 
            txt = f"Reta de Ajuste: y = {round(A, 3)}x + {round(B, 3)}, RMSE = {round(rmse, 3)}, R²={round(r2, 3)}"
        else: 
            txt = f"Reta de Ajuste: y = {round(A, 3)}x {round(B, 3)}, RMSE = {round(rmse, 3)}, R²={round(r2, 3)}"
        plt.text(x.min(), y.min(), txt, fontsize=10, color='#282828')
        plt.legend().set_title("Dados")
        # Exibindo o gráfico
        plt.grid(True)
        plt.show()

    '''
    def tabela(self):
        """
        Retorna um dataframe com os dados originais e previsões
        """
        table = pd.DataFrame({
            'x_original': [ponto[0] for ponto in self.pontos], 
            'y_original': [ponto[1] for ponto in self.pontos],
            'y_ajuste': [ponto[1] for ponto in self.pontos_regressao], 
        })
        return table
    '''


