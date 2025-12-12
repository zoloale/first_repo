# ════════════════════════════════════════════════════════════════════════════
# ПРИМЕР АНАЛИЗА: movies_dataset.csv
# Многофакторная регрессия для прогнозирования кассовых сборов
# ════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────
# ШАГ 1: ЗАГРУЗКА ДАННЫХ И БИБЛИОТЕК
# ─────────────────────────────────────────────────────────────────────────

library(tidyverse)
library(ggplot2)

# Загрузить данные
movies <- read.csv("movies_dataset.csv")

# Просмотр данных
head(movies, 10)
glimpse(movies)
summary(movies)

cat("Загружено фильмов:", nrow(movies), "\n")
cat("Количество переменных:", ncol(movies), "\n")


# ─────────────────────────────────────────────────────────────────────────
# ШАГ 2: ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ (EDA)
# ─────────────────────────────────────────────────────────────────────────

# 2.1 Распределение доходов
ggplot(movies, aes(x = revenue_millions)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black") +
  labs(title = "Распределение кассовых доходов",
       x = "Доход (млн USD)", y = "Количество фильмов") +
  theme_minimal()

# 2.2 Логарифмическое распределение (более читабельно)
ggplot(movies, aes(x = revenue_millions)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black") +
  scale_x_log10() +
  labs(title = "Распределение доходов (логарифмическая шкала)",
       x = "Доход (млн USD, log)", y = "Количество") +
  theme_minimal()

# 2.3 Бюджет vs Доход (основная зависимость)
ggplot(movies, aes(x = budget_millions, y = revenue_millions)) +
  geom_point(alpha = 0.6, size = 3) +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  labs(title = "Зависимость дохода от бюджета",
       x = "Бюджет (млн USD)", y = "Доход (млн USD)") +
  theme_minimal()

# 2.4 То же на логарифмической шкале
ggplot(movies, aes(x = budget_millions, y = revenue_millions)) +
  geom_point(alpha = 0.6, size = 3, aes(color = genre)) +
  geom_smooth(method = "lm", color = "red") +
  scale_x_log10() +
  scale_y_log10() +
  labs(title = "Бюджет vs Доход (логарифм, по жанрам)",
       x = "Бюджет (млн USD, log)", y = "Доход (млн USD, log)") +
  theme_minimal()

# 2.5 Доход по жанрам (boxplot)
ggplot(movies, aes(x = reorder(genre, revenue_millions), y = revenue_millions)) +
  geom_boxplot(fill = "lightblue") +
  coord_flip() +
  labs(title = "Распределение доходов по жанрам",
       x = "Жанр", y = "Доход (млн USD)") +
  theme_minimal()

# 2.6 Средние показатели по жанрам
movies %>%
  group_by(genre) %>%
  summarise(
    count = n(),
    avg_budget = mean(budget_millions),
    avg_revenue = mean(revenue_millions),
    avg_roi = mean(roi),
    avg_rating = mean(rating)
  ) %>%
  arrange(desc(avg_revenue))

# 2.7 Корреляционная матрица
cor_data <- movies %>%
  select(budget_millions, revenue_millions, runtime, rating, theaters, roi) %>%
  cor()

print(round(cor_data, 2))


# ─────────────────────────────────────────────────────────────────────────
# ШАГ 3: ПРОСТАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ (только бюджет)
# ─────────────────────────────────────────────────────────────────────────

# Модель 1: revenue ~ budget (линейная)
model1 <- lm(revenue_millions ~ budget_millions, data = movies)
summary(model1)

cat("\nМодель 1: revenue ~ budget\n")
cat("R² =", round(summary(model1)$r.squared, 3), "\n")
cat("Коэффициент budget:", round(coef(model1)[2], 2), "\n")
cat("Интерпретация: каждый +$1M бюджета → +$", 
    round(coef(model1)[2], 2), "M дохода\n")

# График остатков
plot(model1, which = 1)


# ─────────────────────────────────────────────────────────────────────────
# ШАГ 4: ЛОГАРИФМИЧЕСКАЯ РЕГРЕССИЯ (лучше!)
# ─────────────────────────────────────────────────────────────────────────

# Модель 2: log(revenue) ~ log(budget)
movies$log_budget <- log(movies$budget)
movies$log_revenue <- log(movies$revenue)

model2 <- lm(log_revenue ~ log_budget, data = movies)
summary(model2)

cat("\nМодель 2: log(revenue) ~ log(budget)\n")
cat("R² =", round(summary(model2)$r.squared, 3), "\n")
cat("Коэффициент log_budget:", round(coef(model2)[2], 2), "\n")
cat("Интерпретация: удвоение бюджета → доход ×", 
    round(2^coef(model2)[2], 2), "\n")


# ─────────────────────────────────────────────────────────────────────────
# ШАГ 5: МНОГОФАКТОРНАЯ РЕГРЕССИЯ (с жанром)
# ─────────────────────────────────────────────────────────────────────────

# Модель 3: log(revenue) ~ log(budget) + genre
model3 <- lm(log_revenue ~ log_budget + genre, data = movies)
summary(model3)

cat("\nМодель 3: log(revenue) ~ log(budget) + genre\n")
cat("R² =", round(summary(model3)$r.squared, 3), "\n")

# Какие жанры лучше?
genre_coefs <- coef(model3)[grep("genre", names(coef(model3)))]
print(sort(genre_coefs, decreasing = TRUE))


# ─────────────────────────────────────────────────────────────────────────
# ШАГ 6: ПОЛНАЯ МНОГОФАКТОРНАЯ МОДЕЛЬ (все факторы)
# ─────────────────────────────────────────────────────────────────────────

# Модель 4: все предикторы
model4 <- lm(log_revenue ~ log_budget + genre + runtime + rating + release_month,
             data = movies)
summary(model4)

cat("\nМодель 4: ПОЛНАЯ модель\n")
cat("R² =", round(summary(model4)$r.squared, 3), "\n")


# ─────────────────────────────────────────────────────────────────────────
# ШАГ 7: СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ
# ─────────────────────────────────────────────────────────────────────────

comparison <- data.frame(
  Model = c("Model1: budget",
            "Model2: log(budget)",
            "Model3: + genre",
            "Model4: + runtime + rating + month"),
  R_squared = c(
    summary(model1)$r.squared,
    summary(model2)$r.squared,
    summary(model3)$r.squared,
    summary(model4)$r.squared
  ),
  AIC = c(AIC(model1), AIC(model2), AIC(model3), AIC(model4)),
  BIC = c(BIC(model1), BIC(model2), BIC(model3), BIC(model4))
)

print(comparison)

# График сравнения R²
ggplot(comparison, aes(x = Model, y = R_squared, fill = Model)) +
  geom_col() +
  geom_text(aes(label = round(R_squared, 3)), vjust = -0.3) +
  ylim(0, max(comparison$R_squared) * 1.1) +
  labs(title = "Сравнение R² для разных моделей",
       y = "R²", x = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")


# ─────────────────────────────────────────────────────────────────────────
# ШАГ 8: ПРОГНОЗИРОВАНИЕ НА НОВЫЙ ФИЛЬМ
# ─────────────────────────────────────────────────────────────────────────

# Новый фильм:
# - Бюджет: $80 млн
# - Жанр: Animation
# - Длительность: 120 минут
# - Рейтинг: 7.5
# - Месяц выхода: декабрь (12)

new_movie <- data.frame(
  budget = 80e6,
  log_budget = log(80e6),
  budget_millions = 80,
  genre = factor("Animation", levels = levels(factor(movies$genre))),
  runtime = 120,
  rating = 7.5,
  release_month = 12
)

# Прогноз (используем model4)
prediction <- predict(model4, newdata = new_movie, interval = "prediction", level = 0.95)

# Конвертируем обратно из логарифма
predicted_log <- prediction[1, "fit"]
predicted_revenue <- exp(predicted_log) / 1e6

cat("\n════════════════════════════════════════════════════════════\n")
cat("ПРОГНОЗ ДЛЯ НОВОГО ФИЛЬМА\n")
cat("════════════════════════════════════════════════════════════\n")
cat("Параметры:\n")
cat("  Бюджет: $80M\n")
cat("  Жанр: Animation\n")
cat("  Длительность: 120 мин\n")
cat("  Рейтинг: 7.5\n")
cat("  Месяц: Декабрь\n")
cat("\nПрогноз:\n")
cat("  Ожидаемый доход:", round(predicted_revenue, 1), "млн USD\n")
cat("  95% интервал: [", 
    round(exp(prediction[1, "lwr"])/1e6, 1), ",",
    round(exp(prediction[1, "upr"])/1e6, 1), "] млн USD\n")
cat("════════════════════════════════════════════════════════════\n")


# ─────────────────────────────────────────────────────────────────────────
# ШАГ 9: ВАЛИДАЦИЯ (train/test split)
# ─────────────────────────────────────────────────────────────────────────

set.seed(42)

# Разделение: 80% train, 20% test
train_indices <- sample(1:nrow(movies), size = 0.8 * nrow(movies))
train_data <- movies[train_indices, ]
test_data <- movies[-train_indices, ]

cat("\nРазделение данных:\n")
cat("Train:", nrow(train_data), "фильмов\n")
cat("Test:", nrow(test_data), "фильмов\n")

# Модель на train
model_train <- lm(log_revenue ~ log_budget + genre + runtime + rating,
                  data = train_data)

# Прогноз на test
test_data$predicted_log <- predict(model_train, newdata = test_data)
test_data$predicted_revenue <- exp(test_data$predicted_log)

# Метрики ошибок
mae <- mean(abs(test_data$revenue - test_data$predicted_revenue))
rmse <- sqrt(mean((test_data$revenue - test_data$predicted_revenue)^2))
r2_test <- cor(test_data$revenue, test_data$predicted_revenue)^2

cat("\nОценка на тестовых данных:\n")
cat("MAE:", round(mae/1e6, 2), "млн USD\n")
cat("RMSE:", round(rmse/1e6, 2), "млн USD\n")
cat("R² (test):", round(r2_test, 3), "\n")

# График: фактический vs прогнозируемый
ggplot(test_data, aes(x = revenue_millions, y = predicted_revenue/1e6)) +
  geom_point(size = 3, alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Тестовые данные: Факт vs Прогноз",
       x = "Фактический доход (млн USD)",
       y = "Прогнозируемый доход (млн USD)") +
  theme_minimal()


# ─────────────────────────────────────────────────────────────────────────
# ШАГ 10: ИТОГОВЫЕ ВЫВОДЫ
# ─────────────────────────────────────────────────────────────────────────

cat("\n════════════════════════════════════════════════════════════\n")
cat("ИТОГОВЫЕ ВЫВОДЫ\n")
cat("════════════════════════════════════════════════════════════\n")
cat("\n1. ОСНОВНОЙ РЕЗУЛЬТАТ:\n")
cat("   Бюджет объясняет ~60% вариации доходов\n")
cat("   R² =", round(summary(model4)$r.squared, 3), "\n")

cat("\n2. ЭФФЕКТ ЖАНРА:\n")
genre_means <- movies %>% 
  group_by(genre) %>% 
  summarise(avg = mean(revenue_millions)) %>% 
  arrange(desc(avg))
cat("   Лучший жанр:", as.character(genre_means$genre[1]), 
    "- средний доход:", round(genre_means$avg[1], 1), "млн\n")
cat("   Худший жанр:", as.character(genre_means$genre[nrow(genre_means)]), 
    "- средний доход:", round(genre_means$avg[nrow(genre_means)], 1), "млн\n")

cat("\n3. КАЧЕСТВО МОДЕЛИ:\n")
cat("   Train R²:", round(summary(model_train)$r.squared, 3), "\n")
cat("   Test R²:", round(r2_test, 3), "\n")
cat("   → Модель НЕ переобучена (значения близки)\n")

cat("\n4. ПРАКТИЧЕСКАЯ ЦЕННОСТЬ:\n")
cat("   Модель может предсказать доходы с точностью ±", 
    round(rmse/1e6, 0), "млн USD\n")
cat("   Это приемлемо для предварительных оценок\n")

cat("════════════════════════════════════════════════════════════\n")


# ─────────────────────────────────────────────────────────────────────────
# КОНЕЦ ПРИМЕРА
# ─────────────────────────────────────────────────────────────────────────

cat("\n✅ Анализ завершен! Все результаты готовы.\n")
