import matplotlib.pyplot as plt
from numpy import sqrt
from math import exp

# Константы
q = 1.6e-19
k = 1.38e-23 / q


# Класс кремния с заданными параметрами
class Si:

    def __init__(self):
        # Si parameters
        self.m_dn = 1.08
        self.m_dp = 0.59
        self.epsilon = 11.7
        self.mu_n_300 = 1350
        self.mu_p_300 = 450

    def Eg(self, temperature):
        return 1.17 - 4.73e-4 / (temperature + 636) * temperature ** 2

    def Ei(self, temperature):
        return self.Eg(temperature) / 2 - k * temperature / 4

    def Cminus(self, temperature):
        return exp((self.Ei(temperature) + 0.57 - self.Eg(temperature)) / k / temperature)

    def Cplus(self, temperature):
        return exp((0.35 - self.Ei(temperature)) / k / temperature)

    def Cdoubleminus(self, temperature):
        return exp((2 * self.Ei(temperature) + 1.25 - 3 * self.Eg(temperature)) / k / temperature)

    # Подстроечный параметр
    def Vl(self, temperature):
        return 2620 * exp(-1.1 / k / temperature)

    # Подстроечный параметр, см^0.66
    def Vp(self, temperature):
        return 9.63e-16 * exp(2.83 / k / temperature)

    # нормализованная концентрация вакансий у границы раздела:
    def Vn(self, temperature, n):
        return (1 + self.Cplus(temperature) * self.ni(temperature) / n + self.Cminus(temperature) * n / self.ni(
            temperature) + self.Cdoubleminus(temperature) * (n / self.ni(temperature)) ** 2) / (
                       1 + self.Cminus(temperature) + self.Cplus(temperature) + self.Cdoubleminus(temperature))

    def Nv(self, temperature):
        return 4.82e15 * (self.m_dp ** 1.5) * (temperature ** 1.5)

    def Nc(self, temperature):
        return 4.82e15 * (self.m_dn ** 1.5) * (temperature ** 1.5)

    def ni(self, temperature):
        return ((self.Nc(temperature) * self.Nv(temperature)) ** 0.5) * exp(
            -1 * self.Eg(temperature) / (2 * temperature * k))

#Функция расчитывающая линейную параболическую константу
def K(A, Ea, temperature):
    return A * exp(- Ea / k / temperature)

#Функция расчитывающая уровень окисления
def X(t, A, B, xi):
    # вместо xi при каждом запуске подставляется предыдущее значение x
    tau = (xi**2+A*xi)/B
    # толщина окисла x(t)
    x = (A/2)*(sqrt(1+(t+tau)/A**2*4*B)-1)
    return x

#Создание объекта кремния для работы с ним
Si = Si()

def oxidation(T, P, time, C2):
    sc = Si
    ni = sc.ni(T)
    
    # Параболическая константа
    # Константы взяты для H2O выше 950 Цельсия
    B = K(A=7, Ea=0.78, temperature=T) * (1 + sc.Vp(T) * ni ** 0.22)*P
    
    # "А" для первого слоя с концентрацией C1 = 1e18
    C1 = 1e18
    A1 = B / (K(A=2.96e6, Ea=2.05, temperature=T) * (1 + sc.Vl(T) * (sc.Vn(T, C1) - 1))) * 1.68/P
    print('A1 = ', A1)
    
    # "А" для следующего слоя с концентрацией C2 = 2e17 (из дано)
    A = B / (K(A=2.96e6, Ea=2.05, temperature=T) * (1 + sc.Vl(T) * (sc.Vn(T, C2) - 1))) * 1.68/P
    print('A = ', A)
    print('B = ', B)
    
    # мкм  Начальная толщина оксила равна нулю т.к. окисление в парах воды
    xi0 = 0
    # мкм  Толщина первого слоя окисла 0,1. 0,45 кремния от превого слоя x0 поглащается поэтому еще делим на 0.45
    xi1 = 0.1/0.45
    n = time * 60
    dt = 1

    # Массивы времени и толщины
    t = []
    x = []
    x1 = []
    t1 = []

    # Присвоение начальных значений для первого слоя
    t1.append(0)
    x1.append(X(0, A1, B, xi0))

    # Цикл расчета толщины окисла для первого слоя
    while x1[-1] <= xi1:
        t1.append(t1[-1] + dt)
        x1.append(X(t1[-1], A1, B, xi0))

    t.append(t1[-1])
    x.append(X(t[-1]-t1[-1], A, B, x1[-1]))

    # Цикл расчета толщины окисла для следующего слоя
    while t[-1] <= n:
        t.append(t[-1] + dt)
        x.append(X(t[-1]-t1[-1], A, B, x1[-1]))

    #Вывод графического представления расчетов
    plt.plot(t, x, c='black')
    plt.plot(t1, x1, c='red')
    plt.plot(t1[-1], x1[-1], c='red', marker='x')
    plt.ylabel('Толщина слоя x, мкм')
    plt.xlabel('Время t, c')
    plt.xlim(0,t[-1])
    plt.ylim(0,x[-1])
    plt.show()

oxidation(1473, 2, 30, 2e17)
