import pulp
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

class Car:
    def __init__(self, name, charging_speed_W, battery_capacity_Wh, commute_distance_km, 
                 battery_usage_Wh_per_km, battery_left_percentage, arrival_hour, departure_hour):
        # Initialize car attributes
        self.name = name
        self.charging_speed_W = charging_speed_W
        self.battery_capacity_Wh = battery_capacity_Wh
        self.commute_distance_km = commute_distance_km
        self.battery_usage_Wh_per_km = battery_usage_Wh_per_km
        self.battery_left_percentage = battery_left_percentage
        self.arrival_hour = arrival_hour
        self.departure_hour = departure_hour
    
    def calculate_charging_need_Wh(self):
        """Calculate how much energy the car needs for the next commute"""
        return self.battery_usage_Wh_per_km * self.commute_distance_km
    
    def calculate_min_charging_time_hours(self):
        """Calculate minimum charging time needed"""
        return self.calculate_charging_need_Wh() / self.charging_speed_W
    
    def calculate_max_charging_time_hours(self):
        """Calculate maximum possible charging time (to fully charge battery)"""
        current_charge_Wh = self.battery_capacity_Wh * (self.battery_left_percentage / 100)
        energy_to_full_Wh = self.battery_capacity_Wh - current_charge_Wh
        return energy_to_full_Wh / self.charging_speed_W
    
    def __repr__(self):
        return f"Car({self.name})"

def optimize_charging_schedule(cars, time_slot_minutes=1, start_hour=0, end_hour=24):
    # Create a linear programming problem to optimize charging schedule
    slots_per_hour = 60 // time_slot_minutes  # Number of slots per hour
    total_slots = (end_hour - start_hour) * slots_per_hour  # Total number of slots in the day
    prob = pulp.LpProblem("EV_Charging_Schedule", pulp.LpMinimize)  # Define the optimization problem
    x = {}  # Decision variables for whether a car is charging in a specific slot
    
    # Define decision variables for each car and time slot
    for car in cars:
        x[car] = {}
        for slot in range(total_slots):
            slot_hour = start_hour + (slot * time_slot_minutes) / 60
            if slot_hour >= car.arrival_hour and slot_hour < car.departure_hour:
                x[car][slot] = pulp.LpVariable(f"x_{car.name}_{slot}", cat='Binary')  # Binary variable
            else:
                x[car][slot] = 0  # Not allowed to charge outside arrival and departure hours
    
    # Add constraint to ensure battery does not exceed 100%
    for car in cars:
        max_energy_Wh = car.battery_capacity_Wh - (car.battery_capacity_Wh * (car.battery_left_percentage / 100))
        max_slots = max_energy_Wh / (car.charging_speed_W * (time_slot_minutes / 60))
        prob += pulp.lpSum(x[car][slot] for slot in range(total_slots) if isinstance(x[car][slot], pulp.LpVariable)) <= max_slots
    
    # Define transition variables to minimize the number of charging interruptions
    transitions = {}
    for car in cars:
        transitions[car] = {}
        for slot in range(1, total_slots):
            if isinstance(x[car][slot-1], pulp.LpVariable) and isinstance(x[car][slot], pulp.LpVariable):
                transitions[car][slot] = pulp.LpVariable(f"transition_{car.name}_{slot}", cat='Binary')
                prob += transitions[car][slot] >= x[car][slot] - x[car][slot-1]
                prob += transitions[car][slot] >= x[car][slot-1] - x[car][slot]
    
    # Constraint: Only one car can charge in a time slot
    for slot in range(total_slots):
        # Get cars that can charge in this slot
        cars_in_slot = [car for car in cars if isinstance(x[car][slot], pulp.LpVariable)]
        if cars_in_slot:
            # If at least one car can charge in this slot, prefer to use it
            # Instead of <= 1, use = 1 if any car is available
            prob += pulp.lpSum(x[car][slot] for car in cars_in_slot) == 1
    
    # Define the objective function
    objective_components = []
    
    # Maximize minimum charging time for each car
    for car in cars:
        min_slots_needed = int(car.calculate_min_charging_time_hours() * slots_per_hour)
        car_total_slots = pulp.lpSum(x[car][slot] for slot in range(total_slots) if isinstance(x[car][slot], pulp.LpVariable))
        
        min_charging = pulp.LpVariable(f"min_charging_{car.name}", lowBound=0, upBound=min_slots_needed)
        prob += min_charging <= car_total_slots
        objective_components.append(-10000 * min_charging)
    
    # Minimize the number of transitions (charging interruptions)
    for car in cars:
        total_transitions = pulp.lpSum(transitions[car][slot] for slot in transitions[car])
        objective_components.append(100 * total_transitions)
    
    # Calculate extra charging times
    extra_charging_times = {}
    for car in cars:
        car_total_slots = pulp.lpSum(x[car][slot] for slot in range(total_slots) if isinstance(x[car][slot], pulp.LpVariable))
        min_slots_needed = int(car.calculate_min_charging_time_hours() * slots_per_hour)
        extra_charging_times[car] = car_total_slots - min_slots_needed
    
    # Calculate constraints for balancing extra charging time
    first_arrival = min(car.arrival_hour for car in cars)
    last_departure = max(car.departure_hour for car in cars)
    total_time_span = last_departure - first_arrival
    
    # Calculate total time that must be allocated for minimum charging
    total_min_charging_time = sum(car.calculate_min_charging_time_hours() for car in cars)
    
    # Calculate total available charging slots
    total_available_slots = 0
    for slot in range(total_slots):
        slot_hour = start_hour + (slot * time_slot_minutes) / 60
        cars_available = [car for car in cars if car.arrival_hour <= slot_hour < car.departure_hour]
        if cars_available:
            total_available_slots += 1
    
    # Calculate theoretical maximum extra charging time per car
    # (this assumes perfect distribution which might not be possible given constraints)
    total_available_time = total_available_slots * (time_slot_minutes / 60)
    remaining_time = total_available_time - total_min_charging_time
    average_extra_per_car = remaining_time / len(cars) if len(cars) > 0 else 0
    
    # Convert to slots
    average_extra_slots = average_extra_per_car * slots_per_hour
    
    # Create fairness variable to minimize maximum difference from average
    max_deviation = pulp.LpVariable("max_deviation", lowBound=1)
    for car in cars:
        # Constraint: deviation of extra charging from average can't exceed max_deviation
        prob += extra_charging_times[car] - average_extra_slots <= max_deviation
        prob += average_extra_slots - extra_charging_times[car] <= max_deviation
    
    # Add minimizing the maximum deviation to the objective
    objective_components.append(1000 * max_deviation)
    
    # Add utilization objective - maximize total charging time
    total_charging = pulp.lpSum(x[car][slot] for car in cars for slot in range(total_slots) 
                               if isinstance(x[car][slot], pulp.LpVariable))
    objective_components.append(-10 * total_charging)  # Lower priority than balancing
    
    prob += pulp.lpSum(objective_components)  # Combine all objectives
    
    # Solve the optimization problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Extract the charging schedule
    schedule = {car: [] for car in cars}
    if pulp.LpStatus[prob.status] == 'Optimal':
        for car in cars:
            for slot in range(total_slots):
                if isinstance(x[car][slot], pulp.LpVariable) and pulp.value(x[car][slot]) > 0.5:
                    slot_start = start_hour + (slot * time_slot_minutes) / 60
                    schedule[car].append(slot_start)
    
    return schedule

def visualize_schedule(cars, schedule, time_slot_minutes=15, start_hour=0, end_hour=24):
    # Adjust the figure size dynamically based on the number of cars
    fig, ax = plt.subplots(figsize=(12, max(6, len(cars) * 1.5)))
    
    slots_per_hour = 60 // time_slot_minutes
    total_slots = (end_hour - start_hour) * slots_per_hour
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(cars)))  # Assign unique colors to cars
    
    # Draw arrival and departure windows for each car
    for i, car in enumerate(cars):
        ax.add_patch(plt.Rectangle((car.arrival_hour, i-0.4), car.departure_hour - car.arrival_hour, 0.8, 
                                   color=colors[i], alpha=0.1))
    
    # Draw charging blocks for each car
    for i, car in enumerate(cars):
        charging_blocks = []
        current_block = []
        
        sorted_slots = sorted(schedule[car])
        for slot in sorted_slots:
            if not current_block or abs(slot - current_block[-1] - time_slot_minutes/60) < 0.001:
                current_block.append(slot)
            else:
                charging_blocks.append(current_block)
                current_block = [slot]
        if current_block:
            charging_blocks.append(current_block)
        
        min_time_needed = car.calculate_min_charging_time_hours()
        min_slots_needed = int(min_time_needed * slots_per_hour)
        
        # Accumulated hours to track how much we've drawn
        accumulated_hours = 0
            
        for block in charging_blocks:
            block_start = block[0]
            block_end = block[-1] + time_slot_minutes/60
            block_duration = block_end - block_start
            
            # Draw minimum required time with darker color
            min_time_remaining = max(0, min_time_needed - accumulated_hours)
            min_time_in_block = min(block_duration, min_time_remaining)
            
            if min_time_in_block > 0:
                ax.add_patch(plt.Rectangle((block_start, i-0.4), min_time_in_block, 0.8, 
                                         color=colors[i], alpha=0.9))
                
            # Draw extra time with lighter color
            extra_time = block_duration - min_time_in_block
            if extra_time > 0:
                ax.add_patch(plt.Rectangle((block_start + min_time_in_block, i-0.4), extra_time, 0.8, 
                                        color=colors[i], alpha=0.4, hatch='/'))
                
            accumulated_hours += block_duration
    
    ax.set_title('EV Charging Schedule (One Car at a Time)', fontsize=14)
    ax.set_xlabel('Time (hours)', fontsize=12)
    
    ax.set_yticks(range(len(cars)))
    ax.set_yticklabels([])  # Remove y-axis numbers
    
    # for i, car in enumerate(cars):
    #     ax.text(-0.01, i, car.name, ha='right', va='center', fontsize=10, 
    #            transform=ax.get_yaxis_transform())
    
    ax.grid(True, axis='x')
    ax.set_xlim(start_hour - 1, end_hour + 1)  # Add padding to the x-axis
    ax.set_ylim(-1, len(cars))  # Add padding to the y-axis
    
    ax.set_xticks(range(start_hour, end_hour + 1))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(start_hour, end_hour + 1)])
    
    # Create legend with separate entries for minimum and extra time
    min_time_patch = plt.Rectangle((0,0), 1, 1, color='gray', alpha=0.9)
    extra_time_patch = plt.Rectangle((0,0), 1, 1, color='gray', alpha=0.4, hatch='/')
    
    handles = []
    labels = []
    
    for i, car in enumerate(cars):
        car_patch = plt.Rectangle((0,0), 1, 1, color=colors[i], alpha=0.7)
        handles.append(car_patch)
        
        min_time_needed = car.calculate_min_charging_time_hours()
        actual_time = len(schedule[car]) * (time_slot_minutes/60)
        
        sorted_slots = sorted(schedule[car])
        blocks = 1
        for j in range(1, len(sorted_slots)):
            if abs(sorted_slots[j] - sorted_slots[j-1] - time_slot_minutes/60) > 0.001:
                blocks += 1
        
        if len(sorted_slots) == 0:
            blocks = 0
            
        labels.append(f"{car.name}: {actual_time:.1f}h in {blocks} block(s)")
    
    # Add the min/extra time legend entries at the end
    handles.append(min_time_patch)
    handles.append(extra_time_patch)
    labels.append("Minimum required time")
    labels.append("Extra charging time")
    
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.25), 
             ncol=min(3, len(cars) + 2), fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to show top and bottom fully
    return fig

def analyze_schedule(cars, schedule, time_slot_minutes=15):
    print("Optimized Charging Schedule (One Car at a Time):")
    
    total_min_time_needed = sum(car.calculate_min_charging_time_hours() for car in cars)
    total_time_allocated = sum(len(schedule[car]) * (time_slot_minutes/60) for car in cars)
    
    print(f"Overall statistics:")
    print(f"  - Total minimum charging time needed: {total_min_time_needed:.2f} hours")
    print(f"  - Total charging time allocated: {total_time_allocated:.2f} hours")
    print(f"  - Coverage percentage: {(total_time_allocated/total_min_time_needed*100) if total_min_time_needed > 0 else 0:.1f}%")
    print()
    
    for car in cars:
        min_time_needed = car.calculate_min_charging_time_hours()
        actual_slots = len(schedule[car])
        actual_hours = actual_slots * (time_slot_minutes/60)
        
        sorted_slots = sorted(schedule[car])
        charging_blocks = []
        current_block = []
        
        for slot in sorted_slots:
            if not current_block or abs(slot - current_block[-1] - time_slot_minutes/60) < 0.001:
                current_block.append(slot)
            else:
                charging_blocks.append(current_block)
                current_block = [slot]
        if current_block:
            charging_blocks.append(current_block)
        
        print(f"{car.name}:")
        print(f"  - Minimum charging time needed: {min_time_needed:.2f} hours")
        print(f"  - Allocated charging time: {actual_hours:.2f} hours")
        print(f"  - Extra charging time: {max(0, actual_hours - min_time_needed):.2f} hours")
        print(f"  - Coverage percentage: {(actual_hours/min_time_needed*100) if min_time_needed > 0 else 0:.1f}%")
        print(f"  - Number of continuous charging blocks: {len(charging_blocks)}")
        
        for i, block in enumerate(charging_blocks):
            block_start = block[0]
            block_end = block[-1] + time_slot_minutes/60
            print(f"    * Block {i+1}: {block_start:.2f}h to {block_end:.2f}h " +
                  f"({(block_end-block_start):.2f} hours)")
        
        print()

if __name__ == "__main__":
    cars = [
        Car("Tesla Model 3", charging_speed_W=7000, battery_capacity_Wh=60000, 
            commute_distance_km=50, battery_usage_Wh_per_km=160, 
            battery_left_percentage=40, arrival_hour=8, departure_hour=16),
        
        Car("Nissan Leaf", charging_speed_W=6000, battery_capacity_Wh=40000, 
            commute_distance_km=30, battery_usage_Wh_per_km=150, 
            battery_left_percentage=50, arrival_hour=9, departure_hour=17),
        
        Car("Chevy Bolt", charging_speed_W=7500, battery_capacity_Wh=66000, 
            commute_distance_km=70, battery_usage_Wh_per_km=175, 
            battery_left_percentage=30, arrival_hour=7.5, departure_hour=15.5),
            
        Car("VW ID.4", charging_speed_W=11000, battery_capacity_Wh=77000, 
            commute_distance_km=60, battery_usage_Wh_per_km=180, 
            battery_left_percentage=35, arrival_hour=10, departure_hour=18),
    ]
    
    schedule = optimize_charging_schedule(cars, time_slot_minutes=15)
    
    analyze_schedule(cars, schedule)
    
    for car in cars:
        total_charging_time_hours = len(schedule[car]) * (15 / 60)
        charged_energy_Wh = total_charging_time_hours * car.charging_speed_W
        initial_charge_Wh = car.battery_capacity_Wh * (car.battery_left_percentage / 100)
        final_charge_percentage = ((initial_charge_Wh + charged_energy_Wh) / car.battery_capacity_Wh) * 100
        print(f"{car.name} initial charge percentage: {car.battery_left_percentage:.2f}%, final charge percentage: {final_charge_percentage:.2f}%")
    
    fig = visualize_schedule(cars, schedule)
    plt.show()