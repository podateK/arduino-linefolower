#include <QTRSensors.h>
#include <math.h>

#define NUM_SENSORS 6
#define NUM_HIDDEN_NODES 4
#define NUM_OUTPUT_NODES 2
#define MAX_SPEED 255

QTRSensors qtr;
uint16_t sensorValues[NUM_SENSORS];

float inputLayer[NUM_SENSORS];
float hiddenLayer[NUM_HIDDEN_NODES];
float outputLayer[NUM_OUTPUT_NODES];

float weightsInputHidden[NUM_SENSORS][NUM_HIDDEN_NODES];
float weightsHiddenOutput[NUM_HIDDEN_NODES][NUM_OUTPUT_NODES];

float kp = 0.6, kd = 0.15;
float error = 0, lastError = 0;
int leftMotorSpeed = 0, rightMotorSpeed = 0;

int pwm_a = 9;  // Left motor PWM
int pwm_b = 10; // Right motor PWM
int dir_a = 12; // Left motor direction
int dir_b = 13; // Right motor direction

void setup() {
  qtr.setTypeRC();
  qtr.setSensorPins((const uint8_t[]){2, 3, 4, 5, 6, 7}, NUM_SENSORS);
  qtr.setEmitterPin(13);

  pinMode(pwm_a, OUTPUT);
  pinMode(pwm_b, OUTPUT);
  pinMode(dir_a, OUTPUT);
  pinMode(dir_b, OUTPUT);

  analogWrite(pwm_a, 0);
  analogWrite(pwm_b, 0);

  randomizeWeights();
  calibrateSensors();
  trainNetwork();
}

void loop() {
  qtr.read(sensorValues);

  for (int i = 0; i < NUM_SENSORS; i++) {
    inputLayer[i] = (float)sensorValues[i] / 1000.0;
  }

  feedForward();

  leftMotorSpeed = map(outputLayer[0] * MAX_SPEED, -MAX_SPEED, MAX_SPEED, 0, MAX_SPEED);
  rightMotorSpeed = map(outputLayer[1] * MAX_SPEED, -MAX_SPEED, MAX_SPEED, 0, MAX_SPEED);

  analogWrite(pwm_a, constrain(leftMotorSpeed, 0, MAX_SPEED));
  analogWrite(pwm_b, constrain(rightMotorSpeed, 0, MAX_SPEED));
}

void calibrateSensors() {
  for (int i = 0; i <= 300; i++) {
    qtr.calibrate();
    delay(10);
  }
}

void trainNetwork() {
  for (int epoch = 0; epoch < 1000; epoch++) {
    qtr.read(sensorValues);
    float targetLeft = 0.5, targetRight = 0.5; // Placeholder targets
    float linePosition = qtr.readLineWhite(sensorValues);
    float normalizedPosition = (linePosition - 3500.0) / 3500.0;

    targetLeft = fmax(0, 1.0 - normalizedPosition);
    targetRight = fmax(0, 1.0 + normalizedPosition);

    for (int i = 0; i < NUM_SENSORS; i++) {
      inputLayer[i] = (float)sensorValues[i] / 1000.0;
    }

    feedForward();

    float errorLeft = targetLeft - outputLayer[0];
    float errorRight = targetRight - outputLayer[1];

    backPropagate(errorLeft, errorRight);
  }
}

void randomizeWeights() {
  for (int i = 0; i < NUM_SENSORS; i++) {
    for (int j = 0; j < NUM_HIDDEN_NODES; j++) {
      weightsInputHidden[i][j] = random(-100, 100) / 100.0;
    }
  }

  for (int j = 0; j < NUM_HIDDEN_NODES; j++) {
    for (int k = 0; k < NUM_OUTPUT_NODES; k++) {
      weightsHiddenOutput[j][k] = random(-100, 100) / 100.0;
    }
  }
}

void feedForward() {
  for (int j = 0; j < NUM_HIDDEN_NODES; j++) {
    hiddenLayer[j] = 0;
    for (int i = 0; i < NUM_SENSORS; i++) {
      hiddenLayer[j] += inputLayer[i] * weightsInputHidden[i][j];
    }
    hiddenLayer[j] = tanh(hiddenLayer[j]);
  }

  for (int k = 0; k < NUM_OUTPUT_NODES; k++) {
    outputLayer[k] = 0;
    for (int j = 0; j < NUM_HIDDEN_NODES; j++) {
      outputLayer[k] += hiddenLayer[j] * weightsHiddenOutput[j][k];
    }
    outputLayer[k] = tanh(outputLayer[k]);
  }
}

void backPropagate(float errorLeft, float errorRight) {
  float deltaOutput[NUM_OUTPUT_NODES];
  float deltaHidden[NUM_HIDDEN_NODES];

  deltaOutput[0] = errorLeft * (1 - pow(outputLayer[0], 2));
  deltaOutput[1] = errorRight * (1 - pow(outputLayer[1], 2));

  for (int j = 0; j < NUM_HIDDEN_NODES; j++) {
    deltaHidden[j] = 0;
    for (int k = 0; k < NUM_OUTPUT_NODES; k++) {
      deltaHidden[j] += deltaOutput[k] * weightsHiddenOutput[j][k];
    }
    deltaHidden[j] *= (1 - pow(hiddenLayer[j], 2));
  }

  for (int j = 0; j < NUM_HIDDEN_NODES; j++) {
    for (int k = 0; k < NUM_OUTPUT_NODES; k++) {
      weightsHiddenOutput[j][k] += 0.1 * deltaOutput[k] * hiddenLayer[j];
    }
  }

  for (int i = 0; i < NUM_SENSORS; i++) {
    for (int j = 0; j < NUM_HIDDEN_NODES; j++) {
      weightsInputHidden[i][j] += 0.1 * deltaHidden[j] * inputLayer[i];
    }
  }
}
