// Define motor control pins for motor A
#define L298N_enA 9   // PWM for motor A
#define L298N_in1 12  // Direction 1 for motor A
#define L298N_in2 13  // Direction 2 for motor A

// Define motor control pins for motor B
#define L298N_enB 11  // PWM for motor B
#define L298N_in3 7  // Direction 1 for motor B
#define L298N_in4 8   // Direction 2 for motor B

void setup() {
  // Set motor control pins as outputs
  pinMode(L298N_enA, OUTPUT);
  pinMode(L298N_in1, OUTPUT);
  pinMode(L298N_in2, OUTPUT);
  pinMode(L298N_enB, OUTPUT);
  pinMode(L298N_in3, OUTPUT);
  pinMode(L298N_in4, OUTPUT);
  
  // Initialize serial communication
  Serial.begin(9600);
}

void loop() {
  // Run motor A forward
  digitalWrite(L298N_in1, HIGH);
  digitalWrite(L298N_in2, LOW);
  analogWrite(L298N_enA, 1000); // Set PWM speed for motor A (adjust as needed)

  // Run motor B forward
  digitalWrite(L298N_in3, HIGH);
  digitalWrite(L298N_in4, LOW);
  analogWrite(L298N_enB, 1000); // Set PWM speed for motor B (adjust as needed)

  delay(10000); // Run for 10 seconds
  
  // Stop both motors
  analogWrite(L298N_enA, 0);
  analogWrite(L298N_enB, 0);
  delay(1000); // Pause for 1 second
  
  // Run motor A backward
  digitalWrite(L298N_in1, LOW);
  digitalWrite(L298N_in2, HIGH);
  analogWrite(L298N_enA, 1000); // Set PWM speed for motor A (adjust as needed)

  // Run motor B backward
  digitalWrite(L298N_in3, LOW);
  digitalWrite(L298N_in4, HIGH);
  analogWrite(L298N_enB, 1000); // Set PWM speed for motor B (adjust as needed)

  delay(2000); // Run for 2 seconds
  
  // Stop both motors
  analogWrite(L298N_enA, 0);
  analogWrite(L298N_enB, 0);
  delay(1000); // Pause for 1 second
}
