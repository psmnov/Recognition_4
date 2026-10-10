import numpy as np
import matplotlib
from scipy.optimize import minimize

SHOW = False          # поставь True, чтобы график открывался окном
if not SHOW:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =====================================================================
# ИСХОДНЫЕ ДАННЫЕ
# =====================================================================
a1 = np.array([-3.0, 3.0])

# --- Данные для пунктов 1, 2, 4 (рис. 1) ---
X1 = np.array([(-4, 0), (-4, 1), (-4, 2), (-3, -1), (-2, 1), (-1, 0), (0, -1), (2, -2)], dtype=float)
Y1 = np.array([(-4, 4), (-1, 2), (0, 1), (0, 3), (2, 3), (3, -1)], dtype=float)

# --- Данные для пункта 3 (рис. 2) ---
X3 = np.array([(0, 3), (0, 4), (0, 5), (1, 2), (2, 4), (3, 3), (4, 2), (6, 1)], dtype=float)
Y3 = np.array([(0, 7), (3, 5), (4, 4), (4, 6), (6, 6), (7, 2)], dtype=float)

# Имена объектов: в п.4 вероятности заданы для x1..x7, x0 и y1..y5, y0,
# поэтому ПОСЛЕДНИЙ объект каждого списка - это x0 и y0 (допущение)
NAMES_X = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x0']
NAMES_Y = ['y1', 'y2', 'y3', 'y4', 'y5', 'y0']


def build_Z(X, Y):
    """Z = X U (-Y): объекты Y берём с обратным знаком,
    чтобы все условия имели вид (a, z) > 0."""
    Z = np.vstack((X, -Y))
    names = NAMES_X + ['-' + n for n in NAMES_Y]
    return Z, names


# =====================================================================
# ПУНКТ 1. Метод постоянных приращений
# =====================================================================
def fixed_increment(Z, names, a_start, max_corrections=100000):
    """Циклически обходим Z. Если (a, z) <= 0 -> a = a + z (приращение постоянное: 1*z).
    Останавливаемся, когда за полный проход не было ни одной коррекции."""
    a = a_start.copy()
    k = 0                      # число коррекций (= номер итерации)
    log = []
    while True:
        corrected = False
        for z, name in zip(Z, names):
            if a @ z <= 0:
                a = a + z
                k += 1
                corrected = True
                log.append((k, name, z.copy(), a.copy()))
                if k > max_corrections:
                    raise RuntimeError('не сошёлся')
        if not corrected:
            break
    return a, k, log


def perceptron_bound(Z, a_start, a_hat):
    """Теоретическая оценка максимального числа коррекций (теорема о сходимости персептрона).

    Пусть a* - любой вектор с (a*, z) >= beta2 для всех z, где beta2 = max ||z||^2.
    На каждой коррекции:
        ||a_{k+1} - a*||^2 = ||a_k - a*||^2 + 2(a_k,z) - 2(a*,z) + ||z||^2
                          <= ||a_k - a*||^2 - 2*beta2 + beta2 = ||a_k - a*||^2 - beta2
    (использовано (a_k, z) <= 0 - иначе коррекции не было бы).
    Квадрат нормы не может стать отрицательным, поэтому  k * beta2 <= ||a_1 - a*||^2, т.е.

        k <= ||a_1 - a*||^2 / beta2.

    Считаем две версии оценки:
      (А) a* = alpha * a_hat, alpha = beta2/gamma, gamma = min (a_hat, z),
          где a_hat - найденный алгоритмом разделяющий вектор;
      (Б) a* подбирается так, чтобы оценка была наименьшей (задача на минимум
          ||a_1 - a*||^2 при ограничениях Z a* >= beta2)."""
    beta2 = np.max(np.sum(Z ** 2, axis=1))
    gamma = np.min(Z @ a_hat)
    alpha = beta2 / gamma
    a_star_A = alpha * a_hat
    bound_A = np.sum((a_start - a_star_A) ** 2) / beta2

    res = minimize(lambda a: np.sum((a_start - a) ** 2), x0=a_star_A,
                   constraints=[{'type': 'ineq', 'fun': lambda a: Z @ a - beta2}],
                   method='SLSQP')
    a_star_B = res.x
    bound_B = res.fun / beta2
    return beta2, gamma, alpha, a_star_A, bound_A, a_star_B, bound_B


# =====================================================================
# ПУНКТ 2. Метод переменных приращений (абсолютная коррекция)
# =====================================================================
def variable_increment(Z, names, a_start, margin=1.0, max_corrections=100000):
    """Если (a, z) <= 0 -> a = a + rho*z, где rho = (-(a,z) + margin) / ||z||^2. //ПОЧЕМУ ИМЕННО ТАКОЕ РО И ТАКОЙ МАРЖИН ОБОСНОВАТЬ
    Тогда сразу после коррекции (a_new, z) = margin > 0:
    ошибочный объект исправляется за ОДНУ коррекцию."""
    a = a_start.copy()
    k = 0
    log = []
    while True:
        corrected = False
        for z, name in zip(Z, names):
            s = a @ z
            if s <= 0:
                rho = (-s + margin) / (z @ z)
                a = a + rho * z
                k += 1
                corrected = True
                log.append((k, name, z.copy(), rho, a.copy()))
                if k > max_corrections:
                    raise RuntimeError('не сошёлся')
        if not corrected:
            break
    return a, k, log


# =====================================================================
# ПУНКТ 3. Постоянные приращения с порогом d: (a,y) > d, (a,x) < d
# =====================================================================
def fixed_increment_threshold(X, Y, a_start, d, max_corrections=100000):
    """Если (a, y) <= d  -> a = a + y   (y должен оказаться ВЫШЕ порога)
       Если (a, x) >= d  -> a = a - x   (x должен оказаться НИЖЕ порога)"""
    a = a_start.copy()
    k = 0
    log = []
    seen = set()                      # вектора a в конце каждого прохода: повтор = цикл
    while True:
        corrected = False
        for y, name in zip(Y, NAMES_Y):
            if a @ y <= d:
                a = a + y
                k += 1
                corrected = True
                log.append((k, name, 'a+y', a.copy()))
        for x, name in zip(X, NAMES_X):
            if a @ x >= d:
                a = a - x
                k += 1
                corrected = True
                log.append((k, name, 'a-x', a.copy()))
        if not corrected:
            break
        key = tuple(a)
        if key in seen or k > max_corrections:
            return a, None, []        # алгоритм зациклился - не сходится при этом d
        seen.add(key)
    return a, k, log


# =====================================================================
# ПУНКТ 4. Стохастическая аппроксимация
# =====================================================================
P_X, P_Y = 4 / 7, 3 / 7
P_x_in_X = np.array([0.06, 0.10, 0.13, 0.04, 0.18, 0.16, 0.14, 0.19])   # x1..x7, x0
P_y_in_Y = np.array([0.08, 0.18, 0.22, 0.12, 0.18, 0.22])               # y1..y5, y0


def stochastic_approximation(Z, names, probs, a_start, c=1.0, seed=0, max_iter=200000):
    """На шаге k случайно выбираем z_k с вероятностью probs.
    Если (a, z_k) <= 0 -> a = a + rho_k * z_k, rho_k = c / k.
    (ряд rho_k расходится, а ряд rho_k^2 сходится - условия Роббинса-Монро).
    Останов: все (a, z) > 0."""
    rng = np.random.default_rng(seed)
    a = a_start.copy()
    corrections = 0
    for k in range(1, max_iter + 1):
        i = rng.choice(len(Z), p=probs)
        z = Z[i]
        if a @ z <= 0:
            a = a + (c / k) * z
            corrections += 1
        if np.all(Z @ a > 0):
            return a, k, corrections
    return a, None, corrections


# =====================================================================
# ГРАФИКА
# =====================================================================
def plot_result(ax, X, Y, a_final, a_start, title, d=0.0):
    ax.scatter(X[:, 0], X[:, 1], color='tab:blue', s=50, edgecolor='black', label='X', zorder=3)
    ax.scatter(Y[:, 0], Y[:, 1], color='tab:red', s=50, edgecolor='black', label='Y', zorder=3)

    xs = np.linspace(-6, 9, 200)
    for a, style, color, lab in [(a_start, '--', 'gray', f'старт a1=({a_start[0]:.0f},{a_start[1]:.0f})'),
                                 (a_final, '-', 'green',
                                  f'a=({a_final[0]:.3f}, {a_final[1]:.3f})')]:
        if abs(a[1]) > 1e-12:
            ax.plot(xs, (d - a[0] * xs) / a[1], style, color=color, linewidth=2, label=lab)
    ax.set_xlim(-6, 9 if d else 4)
    allp = np.vstack((X, Y))
    ax.set_ylim(allp[:, 1].min() - 2, allp[:, 1].max() + 2)
    ax.set_xlim(allp[:, 0].min() - 2, allp[:, 0].max() + 2)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)


# =====================================================================
# ОСНОВНАЯ ПРОГРАММА
# =====================================================================
if __name__ == '__main__':
    Z1, names1 = build_Z(X1, Y1)
    print('Z = X U (-Y):')
    for n, z in zip(names1, Z1):
        print(f'  {n:>4} = {z}')

    # ---------------- ПУНКТ 1 ----------------
    print('\n' + '=' * 70)
    print('ПУНКТ 1. Метод постоянных приращений')
    print('=' * 70)
    a_fixed, k_fixed, log1 = fixed_increment(Z1, names1, a1)
    print(f'a_1 = {a1}')
    for k, name, z, a_new in log1:
        print(f'  k={k:>3}: коррекция по {name:>4}={z}  ->  a_{k + 1} = {a_new}')
    print(f'Итого коррекций k = {k_fixed},  разделяющий вектор a = {a_fixed}')
    print('Проверка (a, z) для всех z:', Z1 @ a_fixed, ' -> все > 0:', bool(np.all(Z1 @ a_fixed > 0)))

    beta2, gamma, alpha, a_star_A, bound_A, a_star_B, bound_B = perceptron_bound(Z1, a1, a_fixed)
    print('\nТеоретическая оценка максимального числа коррекций:')
    print(f'  beta^2 = max||z||^2 = {beta2}')
    print(f'  gamma = min(a_hat, z) = {gamma:.4f}, alpha = beta^2/gamma = {alpha:.4f}')
    print(f'  (А) a* = alpha*a_hat = {a_star_A},  k <= ||a1-a*||^2/beta^2 = {bound_A:.2f}  -> k <= {int(np.floor(bound_A))}')
    print(f'  (Б) оптимальный a* = {a_star_B},  k <= {bound_B:.2f}  -> k <= {int(np.floor(bound_B))}')
    print(f'  Фактическое k = {k_fixed}  (оценка - верхняя граница, а не точное значение)')

    # ---------------- ПУНКТ 2 ----------------
    print('\n' + '=' * 70)
    print('ПУНКТ 2. Метод переменных приращений (абсолютная коррекция)')
    print('=' * 70)
    a_var, k_var, log2 = variable_increment(Z1, names1, a1)
    for k, name, z, rho, a_new in log2:
        print(f'  k={k:>3}: {name:>4}={z}, rho={rho:.4f}  ->  a_{k + 1} = {a_new}')
    print(f'Итого коррекций k = {k_var},  разделяющий вектор a = {a_var}')
    print('Проверка (a, z):', Z1 @ a_var, ' -> все > 0:', bool(np.all(Z1 @ a_var > 0)))

    # ---------------- ПУНКТ 3 ----------------
    print('\n' + '=' * 70)
    print('ПУНКТ 3. Постоянные приращения с порогом d')
    print('=' * 70)
    print('В условии значение d не задано. Метод с фиксированным порогом не гарантирует')
    print('сходимость (теорема о персептроне доказана для d=0), поэтому проверяем d = 1..40:')
    results3 = {}
    for d in range(1, 41):
        results3[d] = fixed_increment_threshold(X3, Y3, a1, d)
    conv = [d for d, r in results3.items() if r[1] is not None]
    print('  сходится при d =', conv)
    print('  зацикливается при d =', [d for d in results3 if d not in conv])

    D_CHOSEN = 17     # <-- подставь d из своего задания (наименьшее d, при котором метод сходится)
    a_thr, k_thr, log3 = results3[D_CHOSEN]
    if k_thr is None:
        raise SystemExit(f'При d={D_CHOSEN} метод зацикливается - выбери другое d из списка выше')
    print(f'\nПодробно для d = {D_CHOSEN}:')
    for k, name, op, a_new in log3:
        print(f'  k={k:>3}: {name:>3} ({op})  ->  a = {a_new}')
    print(f'k = {k_thr}, a = {a_thr}')
    print('(a, y) > d:', bool(np.all(Y3 @ a_thr > D_CHOSEN)), '   (a, x) < d:', bool(np.all(X3 @ a_thr < D_CHOSEN)))

    # ---------------- ПУНКТ 4 ----------------
    print('\n' + '=' * 70)
    print('ПУНКТ 4. Стохастическая аппроксимация')
    print('=' * 70)
    probs = np.concatenate((P_X * P_x_in_X, P_Y * P_y_in_Y))
    print('Вероятности выбора объектов z (P(класс) * P(объект|класс)):')
    for n, p in zip(names1, probs):
        print(f'  P({n:>4}) = {p:.4f}')
    print('Сумма вероятностей =', probs.sum())

    a_sa, k_sa, corr_sa = stochastic_approximation(Z1, names1, probs, a1, c=1.0, seed=0)
    print(f'\nОдин прогон (seed=0, rho_k = 1/k): итераций (выборов z) = {k_sa}, из них коррекций = {corr_sa}')
    print('a =', a_sa, ' (a, z):', Z1 @ a_sa)

    iters = []
    for s in range(200):
        _, kk, _ = stochastic_approximation(Z1, names1, probs, a1, c=1.0, seed=s)
        iters.append(kk)
    iters = [i for i in iters if i is not None]
    print(f'По 200 прогонам с разными seed: сошлось {len(iters)}/200, '
          f'число итераций: min={min(iters)}, медиана={int(np.median(iters))}, max={max(iters)}')

    # ---------------- ГРАФИКИ ----------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    plot_result(axes[0, 0], X1, Y1, a_fixed, a1, f'П.1. Постоянные приращения (k={k_fixed})')
    plot_result(axes[0, 1], X1, Y1, a_var, a1, f'П.2. Переменные приращения (k={k_var})')
    plot_result(axes[1, 0], X3, Y3, a_thr, a1, f'П.3. С порогом d={D_CHOSEN} (k={k_thr})', d=D_CHOSEN)
    plot_result(axes[1, 1], X1, Y1, a_sa, a1, f'П.4. Стохастическая аппроксимация (k={k_sa})')
    plt.tight_layout()
    plt.savefig('lab3_perceptron.png', dpi=150)
    print('\nГрафик сохранён: lab3_perceptron.png')
    if SHOW:
        plt.show()

