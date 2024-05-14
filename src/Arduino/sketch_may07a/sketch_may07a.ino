// Define motor control pins
#define L298N_enA 10  // PWM
#define L298N_in1 2    // Dir Motor A
#define L298N_in2 3    // Dir Motor A

void setup() {
  // Set motor control pins as outputs
  pinMode(L298N_enA, OUTPUT);
  pinMode(L298N_in1, OUTPUT);
  pinMode(L298N_in2, OUTPUT);
  
  // Initialize serial communication
  Serial.begin(9600);
}

void loop() {
  // Run motor forward
  digitalWrite(L298N_in1, HIGH);
  digitalWrite(L298N_in2, LOW);
  analogWrite(L298N_enA, 1000); // Set PWM speed to 100 (adjust as needed)
  delay(10000); // Run for 2 seconds
  
  // Stop motor
  analogWrite(L298N_enA, 0);
  delay(1000); // Pause for 1 second
  
  // Run motor backward
  digitalWrite(L298N_in1, LOW);
  digitalWrite(L298N_in2, HIGH);
  analogWrite(L298N_enA, 1000); // Set PWM speed to 100 (adjust as needed)
  delay(2000); // Run for 2 seconds
  
  // Stop motor
  analogWrite(L298N_enA, 0);
  delay(1000); // Pause for 1 second
}
