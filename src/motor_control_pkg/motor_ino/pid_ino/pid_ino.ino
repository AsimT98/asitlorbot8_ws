#include <Arduino.h>
#include <Encoder.h>
#include <AutoPID.h>

#define right_encoder_phaseA 3  // Interrupt 
#define right_encoder_phaseB 5  
#define left_encoder_phaseA 2   // Interrupt
#define left_encoder_phaseB 4

#define L298N_enA 9    // PWM for motor A
#define L298N_enB 11   // PWM for motor B
#define L298N_in4 8    // Dir Motor B
#define L298N_in3 7    // Dir Motor B
#define L298N_in2 13   // Dir Motor A
#define L298N_in1 12   // Dir Motor A

#define ENCODER_TICKS_PER_REV 1320 // Adjust this according to your encoder specs
#define MAX_SPEED 255  // Maximum PWM value

// PID parameters
double Kp = 22, Ki =  8, Kd = 10; // Initial PID constants
double setpoint = 100; // Target speed
double input_left, output_left;
double input_right, output_right;

Encoder left_encoder(left_encoder_phaseA, left_encoder_phaseB);
Encoder right_encoder(right_encoder_phaseA, right_encoder_phaseB);

// Define AutoPID for left and right motors
AutoPID left_pid(&input_left, &setpoint, &output_left, 0, MAX_SPEED, Kp, Ki, Kd);
AutoPID right_pid(&input_right, &setpoint, &output_right, 0, MAX_SPEED, Kp, Ki, Kd);

void setup() {
  Serial.begin(115200); // Initialize serial communication
  pinMode(L298N_enA, OUTPUT);
  pinMode(L298N_enB, OUTPUT);
  pinMode(L298N_in1, OUTPUT);
  pinMode(L298N_in2, OUTPUT);
  pinMode(L298N_in3, OUTPUT);
  pinMode(L298N_in4, OUTPUT);

  // Initialize AutoPID
  left_pid.setTimeStep(100); // Update interval in ms
  right_pid.setTimeStep(100); // Update interval in ms
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    if (command.startsWith("SETLEFT:")) {
      setpoint = command.substring(8).toInt();
    } else if (command.startsWith("SETRIGHT:")) {
      setpoint = command.substring(9).toInt();
    }
  }

  // Read encoder values
  int left_ticks = left_encoder.read();
  int right_ticks = right_encoder.read();

  // Convert encoder ticks to RPM
  input_left = (left_ticks * 60.0) / (ENCODER_TICKS_PER_REV * 2.0);
  input_right = (right_ticks * 60.0) / (ENCODER_TICKS_PER_REV * 2.0);

  // Run PID computation
  left_pid.run();
  right_pid.run();

  // Send encoder data back to ROS
  Serial.print("LEFT:");
  Serial.println(input_left);
  Serial.print("RIGHT:");
  Serial.println(input_right);

  // Update motor direction and speed
  int left_speed = constrain(abs(output_left), 0, MAX_SPEED);
  int right_speed = constrain(abs(output_right), 0, MAX_SPEED);

  if (output_left >= 0) {
    digitalWrite(L298N_in1, HIGH);
    digitalWrite(L298N_in2, LOW);
  } else {
    digitalWrite(L298N_in1, LOW);
    digitalWrite(L298N_in2, HIGH);
  }

  if (output_right >= 0) {
    digitalWrite(L298N_in3, HIGH);
    digitalWrite(L298N_in4, LOW);
  } else {
    digitalWrite(L298N_in3, LOW);
    digitalWrite(L298N_in4, HIGH);
  }

  analogWrite(L298N_enA, left_speed);
  analogWrite(L298N_enB, right_speed);
}
