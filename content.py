# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np

from utils import polynomial


def mean_squared_error(x, y, w):
    """
    :param x: ciąg wejściowy Nx1
    :param y: ciąg wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: błąd średniokwadratowy pomiędzy wyjściami y oraz wyjściami
     uzyskanymi z wielowamiu o parametrach w dla wejść x
    """

    predictions = polynomial(x,w)


    err = [(y[i] - predictions[i]) ** 2 for i in range(len(y))]
    err = np.float_(sum(err)/len(y))
    return err

def design_matrix(x_train, M):
    """
    :param x_train: ciąg treningowy Nx1
    :param M: stopień wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzędu M
    """


    dm = [[np.float_(x**i) for i in range(M+1)] for x in x_train]
    return np.array(dm)




def least_squares(x_train, y_train, M):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotkę (w,err), gdzie w są parametrami dopasowanego 
    wielomianu, a err to błąd średniokwadratowy dopasowania
    """

    dm = design_matrix(x_train,M)
    w = np.linalg.inv(dm.transpose() @ dm) @ dm.transpose() @ y_train
    err = mean_squared_error(x_train,y_train,w)

    return w,err



def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotkę (w,err), gdzie w są parametrami dopasowanego
    wielomianu zgodnie z kryterium z regularyzacją l2, a err to błąd 
    średniokwadratowy dopasowania
    """
    dm = design_matrix(x_train,M)
    w = np.linalg.inv(dm.transpose() @ dm + regularization_lambda * np.identity(dm.shape[1])) @ dm.transpose() @ y_train
    err = mean_squared_error(x_train,y_train,w)

    return w, err


def model_selection(x_train, y_train, x_val, y_val, M_values):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param x_val: ciąg walidacyjny wejśćia Nx1
    :param y_val: ciąg walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, które mają byc sprawdzone
    :return: funkcja zwraca krotkę (w,train_err,val_err), gdzie w są parametrami
    modelu, ktory najlepiej generalizuje dane, tj. daje najmniejszy błąd na 
    ciągu walidacyjnym, train_err i val_err to błędy na sredniokwadratowe na 
    ciągach treningowym i walidacyjnym
    """

    models = [list(least_squares(x_train,y_train,m)) for m in M_values]

    for i in range(len(models)):
        models[i].append(mean_squared_error(x_val,y_val,models[i][0]))

    model = sorted(models, key=lambda wErr : wErr[2])[0]

    return model


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param x_val: ciąg walidacyjny wejśćia Nx1
    :param y_val: ciąg walidacyjny wyjscia Nx1
    :param M: stopień wielomianu
    :param lambda_values: lista z wartościami różnych parametrów regularyzacji
    :return: funkcja zwraca krotkę (w,train_err,val_err,regularization_lambda),
    gdzie w są parametrami modelu, ktory najlepiej generalizuje dane, tj. daje
    najmniejszy błąd na ciągu walidacyjnym. Wielomian dopasowany jest wg
    kryterium z regularyzacją. train_err i val_err to błędy średniokwadratowe
    na ciągach treningowym i walidacyjnym. regularization_lambda to najlepsza
    wartość parametru regularyzacji
    """
    models = [list(regularized_least_squares(x_train, y_train,M, l)) for l in lambda_values]

    for i in range(len(models)):
        models[i].append(mean_squared_error(x_val, y_val, models[i][0]))
        models[i].append(lambda_values[i])

    model = sorted(models, key=lambda wErr: wErr[2])[0]

    return model