#include <Wire.h>
#include <Adafruit_MLX90640.h>
#include <math.h>

constexpr int SDA_PIN = 8;
constexpr int SCL_PIN = 9;

constexpr uint32_t SERIAL_BAUD = 921600;
constexpr int PIXEL_COUNT = 32 * 24;

constexpr bool SEND_FULL_FRAME = true;

Adafruit_MLX90640 mlx;
float frame[PIXEL_COUNT];
uint32_t frameId = 0;

void setup() {
  Serial.begin(SERIAL_BAUD);

  const uint32_t start = millis();
  while (!Serial && millis() - start < 3000) {
    delay(10);
  }

  Serial.println("#BOOT,MLX90640 data logger");

  if (!Wire.begin(SDA_PIN, SCL_PIN)) {
    Serial.println("#ERROR,I2C initialization failed");
    while (true) delay(1000);
  }

  Wire.setClock(400000);

  if (!mlx.begin(MLX90640_I2CADDR_DEFAULT, &Wire)) {
    Serial.println("#ERROR,MLX90640 initialization failed");
    Serial.println("#ERROR,Check power GND SDA=8 SCL=9");
    while (true) delay(1000);
  }

  mlx.setMode(MLX90640_CHESS);
  mlx.setResolution(MLX90640_ADC_18BIT);
  mlx.setRefreshRate(MLX90640_4_HZ);

  Serial.println("#CONFIG,CHESS,18BIT,REFRESH_4_HZ");

  if (SEND_FULL_FRAME) {
    Serial.println("#MODE,FULL_FRAME");
  } else {
    Serial.println("#MODE,SUMMARY");
    Serial.println("#FIELDS,id,ms,ta,min,max,mean,valid");
  }

  Serial.println("#READY");
}

void loop() {
  const int status = mlx.getFrame(frame);

  if (status != 0) {
    Serial.print("#ERROR,FRAME_READ,");
    Serial.println(status);
    delay(100);
    return;
  }

  // 프레임 읽기가 끝난 시각입니다.
  const uint32_t timestampMs = millis();

  // 새 측정을 하지 않고 가장 최근 프레임의 센서 온도를 사용합니다.
  const float sensorTa = mlx.getTa(false);

  float minTemp = INFINITY;
  float maxTemp = -INFINITY;
  double sum = 0.0;
  int validCount = 0;

  for (int i = 0; i < PIXEL_COUNT; i++) {
    if (!isfinite(frame[i])) continue;

    if (frame[i] < minTemp) minTemp = frame[i];
    if (frame[i] > maxTemp) maxTemp = frame[i];

    sum += frame[i];
    validCount++;
  }

  if (validCount == 0) {
    Serial.println("#ERROR,NO_VALID_PIXELS");
    return;
  }

  if (SEND_FULL_FRAME) {
    // 형식: F,id,ms,ta,pixel_0,...,pixel_767
    Serial.print("F,");
    Serial.print(frameId);
    Serial.print(",");
    Serial.print(timestampMs);
    Serial.print(",");
    Serial.print(sensorTa, 2);

    for (int i = 0; i < PIXEL_COUNT; i++) {
      Serial.print(",");

      if (isfinite(frame[i])) {
        Serial.print(frame[i], 2);
      } else {
        Serial.print("nan");
      }
    }

    Serial.println();
  } else {
    // 형식: #STAT,id,ms,ta,min,max,mean,valid
    Serial.print("#STAT,");
    Serial.print(frameId);
    Serial.print(",");
    Serial.print(timestampMs);
    Serial.print(",");
    Serial.print(sensorTa, 2);
    Serial.print(",");
    Serial.print(minTemp, 2);
    Serial.print(",");
    Serial.print(maxTemp, 2);
    Serial.print(",");
    Serial.print(sum / validCount, 2);
    Serial.print(",");
    Serial.println(validCount);
  }

  frameId++;
}
