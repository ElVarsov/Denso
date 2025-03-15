import tkinter as tk
from tkinter import messagebox
from algorithm import Car, optimize_charging_schedule, analyze_schedule, visualize_schedule
import matplotlib.pyplot as plt

class ChargingSchedulerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EV Charging Scheduler")
        
        self.cars = []
        
        # Input fields for car details
        tk.Label(root, text="Car Name:").grid(row=0, column=0)
        self.name_entry = tk.Entry(root)
        self.name_entry.grid(row=0, column=1)
        
        tk.Label(root, text="Charging Speed (kW):").grid(row=1, column=0)
        self.charging_speed_entry = tk.Entry(root)
        self.charging_speed_entry.grid(row=1, column=1)
        
        tk.Label(root, text="Battery Capacity (kWh):").grid(row=2, column=0)
        self.battery_capacity_entry = tk.Entry(root)
        self.battery_capacity_entry.grid(row=2, column=1)
        
        tk.Label(root, text="Commute Distance (km):").grid(row=3, column=0)
        self.commute_distance_entry = tk.Entry(root)
        self.commute_distance_entry.grid(row=3, column=1)
        
        tk.Label(root, text="Battery Usage (kWh/100 km):").grid(row=4, column=0)
        self.battery_usage_entry = tk.Entry(root)
        self.battery_usage_entry.grid(row=4, column=1)
        
        tk.Label(root, text="Battery Left (%):").grid(row=5, column=0)
        self.battery_left_entry = tk.Entry(root)
        self.battery_left_entry.grid(row=5, column=1)
        
        tk.Label(root, text="Arrival Hour:").grid(row=6, column=0)
        self.arrival_hour_entry = tk.Entry(root)
        self.arrival_hour_entry.grid(row=6, column=1)
        
        tk.Label(root, text="Departure Hour:").grid(row=7, column=0)
        self.departure_hour_entry = tk.Entry(root)
        self.departure_hour_entry.grid(row=7, column=1)
        
        # Buttons
        tk.Button(root, text="Add Car", command=self.add_car).grid(row=8, column=0, pady=10)
        tk.Button(root, text="Optimize Schedule", command=self.optimize_schedule).grid(row=8, column=1, pady=10)
    
    def add_car(self):
        try:
            # Create a Car object from input fields
            car = Car(
                name=self.name_entry.get(),
                charging_speed_W=1000 * int(self.charging_speed_entry.get()),
                battery_capacity_Wh=1000 * int(self.battery_capacity_entry.get()),
                commute_distance_km=float(self.commute_distance_entry.get()),
                battery_usage_Wh_per_km=10 * float(self.battery_usage_entry.get()),
                battery_left_percentage=float(self.battery_left_entry.get()),
                arrival_hour=float(self.arrival_hour_entry.get()),
                departure_hour=float(self.departure_hour_entry.get())
            )
            self.cars.append(car)
            messagebox.showinfo("Success", f"Car '{car.name}' added successfully!")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid values for all fields.")
    
    def optimize_schedule(self):
        if not self.cars:
            messagebox.showerror("Error", "No cars added. Please add cars first.")
            return
        
        # Optimize the charging schedule
        schedule = optimize_charging_schedule(self.cars, time_slot_minutes=15)
        
        # Analyze and display the schedule
        analyze_schedule(self.cars, schedule)
        fig = visualize_schedule(self.cars, schedule)
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = ChargingSchedulerApp(root)
    root.mainloop()
