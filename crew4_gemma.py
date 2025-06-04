import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import speech_recognition as sr
import pyttsx3
import random
import time
from datetime import datetime
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, List, Optional
import json
import re
import os

class MockCabAPI:
    """Mock API for simulating Uber and Ola cab services"""
    
    def __init__(self):
        self.base_prices = {
            'uber': {'base': 50, 'per_km': 12, 'surge_multiplier': 1.0},
            'ola': {'base': 45, 'per_km': 10, 'surge_multiplier': 1.0}
        }
        self.bookings = []
    
    def calculate_distance(self, pickup: str, dropoff: str) -> float:
        """Simulate distance calculation between locations"""
        # Mock distance calculation based on location names
        locations = ['airport', 'mall', 'station', 'hospital', 'office', 'home']
        pickup_clean = pickup.lower().strip()
        dropoff_clean = dropoff.lower().strip()
        
        # Simple mock distance based on string similarity
        base_distance = random.uniform(5, 25)
        if any(loc in pickup_clean for loc in locations) and any(loc in dropoff_clean for loc in locations):
            return round(base_distance, 2)
        return round(base_distance + random.uniform(0, 10), 2)
    
    def get_price(self, platform: str, pickup: str, dropoff: str) -> Dict:
        """Get price estimate for a platform"""
        distance = self.calculate_distance(pickup, dropoff)
        
        if platform.lower() not in self.base_prices:
            return {'error': f'Platform {platform} not supported'}
        
        pricing = self.base_prices[platform.lower()]
        
        # Add some randomness for surge pricing
        surge = random.uniform(1.0, 2.5) if random.random() > 0.7 else 1.0
        
        base_cost = pricing['base'] + (distance * pricing['per_km'])
        final_cost = round(base_cost * surge, 2)
        
        return {
            'platform': platform.title(),
            'price': final_cost,
            'distance': distance,
            'surge_multiplier': round(surge, 2),
            'eta': random.randint(5, 20)
        }
    
    def book_cab(self, platform: str, pickup: str, dropoff: str, price: float) -> Dict:
        """Book a cab on the specified platform"""
        booking_id = f"{platform.upper()}{random.randint(1000, 9999)}"
        booking = {
            'booking_id': booking_id,
            'platform': platform.title(),
            'pickup': pickup,
            'dropoff': dropoff,
            'price': price,
            'status': 'confirmed',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.bookings.append(booking)
        return booking

# Global instance of MockCabAPI to share across tools
mock_cab_api = MockCabAPI()

class PriceRetrievalTool(BaseTool):
    name: str = "price_retrieval"
    description: str = "Get price estimates from Uber and Ola platforms. Input format: 'pickup_location|dropoff_location'"
    
    def _run(self, location_input: str) -> str:
        try:
            if '|' in location_input:
                pickup, dropoff = location_input.split('|', 1)
            else:
                # Try to parse from description
                parts = location_input.split(' to ')
                if len(parts) == 2:
                    pickup, dropoff = parts
                else:
                    return json.dumps({'error': 'Invalid input format. Use: pickup|dropoff'})
            
            pickup = pickup.strip()
            dropoff = dropoff.strip()
            
            uber_price = mock_cab_api.get_price('uber', pickup, dropoff)
            ola_price = mock_cab_api.get_price('ola', pickup, dropoff)
            
            result = {
                'uber': uber_price,
                'ola': ola_price
            }
            return json.dumps(result)
        except Exception as e:
            return json.dumps({'error': str(e)})

class BookingTool(BaseTool):
    name: str = "booking"
    description: str = "Book a cab on the specified platform. Input format: 'platform|pickup|dropoff|price'"
    
    def _run(self, booking_input: str) -> str:
        try:
            parts = booking_input.split('|')
            if len(parts) < 4:
                return json.dumps({'error': 'Invalid input format. Use: platform|pickup|dropoff|price'})
            
            platform = parts[0].strip()
            pickup = parts[1].strip()
            dropoff = parts[2].strip()
            price = float(parts[3].strip())
            
            booking = mock_cab_api.book_cab(platform, pickup, dropoff, price)
            return json.dumps(booking)
        except Exception as e:
            return json.dumps({'error': str(e)})

class SimpleLocationExtractor:
    """Simple rule-based location extractor as fallback"""
    
    def extract_locations(self, text: str) -> Dict[str, str]:
        """Extract pickup and dropoff locations using simple patterns"""
        text = text.lower()
        locations = {'pickup': '', 'dropoff': ''}
        
        # Common patterns
        patterns = [
            r'from\s+(.+?)\s+to\s+(.+?)(?:\s|$|\.)',
            r'pick\s+me\s+up\s+(?:at|from)\s+(.+?)\s+(?:and\s+)?drop\s+(?:me\s+)?(?:at|to)\s+(.+?)(?:\s|$|\.)',
            r'book\s+a\s+cab\s+from\s+(.+?)\s+to\s+(.+?)(?:\s|$|\.)',
            r'go\s+from\s+(.+?)\s+to\s+(.+?)(?:\s|$|\.)',
            r'ride\s+from\s+(.+?)\s+to\s+(.+?)(?:\s|$|\.)',
            r'(.+?)\s+to\s+(.+?)(?:\s|$|\.)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                locations['pickup'] = match.group(1).strip()
                locations['dropoff'] = match.group(2).strip()
                break
        
        return locations

class VoiceCabApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice-Enabled Cab Price Comparison (Gemini)")
        self.root.geometry("900x700")
        
        # Initialize voice components
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 150)
        
        # Initialize fallback location extractor
        self.location_extractor = SimpleLocationExtractor()
        
        # Initialize tools first
        self.price_tool = PriceRetrievalTool()
        self.booking_tool = BookingTool()
        
        # Initialize Gemini and agents
        self.setup_gemini()
        
        # Create UI
        self.create_ui()
        
        # Conversation state
        self.current_pickup = ""
        self.current_dropoff = ""
        self.current_prices = {}
        self.is_listening = False
        
    def setup_gemini(self):
        """Setup Gemini API"""
        self.gemini_api_key = ""
        self.use_llm = False
        self.llm = None
        
        # Initialize agents to None first
        self.voice_agent = None
        self.price_agent = None
        self.comparison_agent = None
        self.booking_agent = None
        
        # Check if API key is already set in environment
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            self.initialize_gemini(api_key)
        else:
            if hasattr(self, 'results_text'):
                self.log_message("Gemini API key not found. Please enter your API key in the settings.")
    
    def initialize_gemini(self, api_key: str):
        """Initialize Gemini with provided API key"""
        try:
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Set environment variable for LiteLLM
            os.environ['GEMINI_API_KEY'] = api_key
            
            # Test the API key with a simple call
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            test_response = model.generate_content("Hello")
            
            # Create LangChain wrapper with correct model specification for CrewAI
            #self.llm = ChatGoogleGenerativeAI(
            #    model="gemini-2.0-flash-exp",  # Use the model name without prefix
            #    google_api_key=api_key,
            #    temperature=0.3
            #)
            
            self.llm=LLM(
                api_key=os.getenv("GEMINI_API_KEY"),
                model="gemini/gemini-2.0-flash",
            )
            
            self.gemini_api_key = api_key
            self.use_llm = True
            self.log_message("Gemini API connected successfully!")
            
            # Setup agents with the new LLM
            self.setup_agents()
            
        except Exception as e:
            self.use_llm = False
            self.log_message(f"Failed to initialize Gemini: {e}")
            self.log_message("Using simple rule-based extraction instead")
    
    def log_message(self, message: str):
        """Add message to results text area"""
        try:
            timestamp = datetime.now().strftime('%H:%M:%S')
            if hasattr(self, 'results_text'):
                self.results_text.insert(tk.END, f"[{timestamp}] {message}\n")
                self.results_text.see(tk.END)
                self.root.update()
            else:
                print(f"[{timestamp}] {message}")
        except:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        
    def setup_agents(self):
        """Setup CrewAI agents for multi-agent architecture"""
        
        if self.use_llm and self.llm:
            try:
                # Agent 1: Voice Processing Agent
                self.voice_agent = Agent(
                    role='Voice Processing Specialist',
                    goal='Process user voice input and extract pickup and dropoff locations',
                    backstory='You are an expert in natural language processing who specializes in extracting location information from user speech.',
                    verbose=False,
                    allow_delegation=False,
                    llm=self.llm
                )
                
                # Agent 2: Price Retrieval Agent
                self.price_agent = Agent(
                    role='Price Retrieval Specialist',
                    goal='Retrieve cab prices from Uber and Ola platforms',
                    backstory='You are a specialist in fetching real-time pricing data from multiple cab booking platforms.',
                    tools=[self.price_tool],
                    verbose=False,
                    allow_delegation=False,
                    llm=self.llm
                )
                
                # Agent 3: Price Comparison Agent
                self.comparison_agent = Agent(
                    role='Price Comparison Analyst',
                    goal='Compare prices between platforms and recommend the best option',
                    backstory='You are an expert analyst who compares prices and provides recommendations to users.',
                    verbose=False,
                    allow_delegation=False,
                    llm=self.llm
                )
                
                # Agent 4: Booking Agent
                self.booking_agent = Agent(
                    role='Booking Specialist',
                    goal='Handle cab bookings on the selected platform',
                    backstory='You are a booking specialist who handles cab reservations efficiently.',
                    tools=[self.booking_tool],
                    verbose=False,
                    allow_delegation=False,
                    llm=self.llm
                )
                
                self.log_message("Agents initialized successfully!")
                
            except Exception as e:
                self.log_message(f"Failed to initialize agents: {e}")
                self.use_llm = False
                # Set agents to None on failure
                self.voice_agent = None
                self.price_agent = None
                self.comparison_agent = None
                self.booking_agent = None
        else:
            # Fallback: no agents will be used
            self.voice_agent = None
            self.price_agent = None
            self.comparison_agent = None
            self.booking_agent = None
    
    def create_ui(self):
        """Create the UI"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Voice-Enabled Cab Price Comparison (Gemini)", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # API Key section
        api_frame = ttk.LabelFrame(main_frame, text="Gemini API Settings", padding="10")
        api_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(api_frame, text="API Key:").grid(row=0, column=0, sticky=tk.W)
        self.api_key_entry = ttk.Entry(api_frame, width=50, show="*")
        self.api_key_entry.grid(row=0, column=1, padx=(10, 10))
        
        self.connect_button = ttk.Button(api_frame, text="Connect", 
                                        command=self.connect_gemini)
        self.connect_button.grid(row=0, column=2)
        
        self.api_status = ttk.Label(api_frame, text="Not connected", foreground="red")
        self.api_status.grid(row=1, column=0, columnspan=3, pady=(5, 0))
        
        # Voice input section
        voice_frame = ttk.LabelFrame(main_frame, text="Voice Input", padding="10")
        voice_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.voice_button = ttk.Button(voice_frame, text="Start Voice Input", 
                                      command=self.toggle_voice_input)
        self.voice_button.grid(row=0, column=0, padx=(0, 10))
        
        self.voice_status = ttk.Label(voice_frame, text="Ready to listen...")
        self.voice_status.grid(row=0, column=1)
        
        # Manual input section
        manual_frame = ttk.LabelFrame(main_frame, text="Manual Input (Optional)", padding="10")
        manual_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(manual_frame, text="Pickup Location:").grid(row=0, column=0, sticky=tk.W)
        self.pickup_entry = ttk.Entry(manual_frame, width=40)
        self.pickup_entry.grid(row=0, column=1, padx=(10, 0))
        
        ttk.Label(manual_frame, text="Dropoff Location:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.dropoff_entry = ttk.Entry(manual_frame, width=40)
        self.dropoff_entry.grid(row=1, column=1, padx=(10, 0), pady=(5, 0))
        
        self.manual_button = ttk.Button(manual_frame, text="Get Prices", 
                                       command=self.process_manual_input)
        self.manual_button.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, width=80)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Booking section
        booking_frame = ttk.LabelFrame(main_frame, text="Booking", padding="10")
        booking_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        self.book_uber_button = ttk.Button(booking_frame, text="Book Uber", 
                                          command=lambda: self.book_cab('uber'), state='disabled')
        self.book_uber_button.grid(row=0, column=0, padx=(0, 10))
        
        self.book_ola_button = ttk.Button(booking_frame, text="Book Ola", 
                                         command=lambda: self.book_cab('ola'), state='disabled')
        self.book_ola_button.grid(row=0, column=1)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Initial status update
        if self.use_llm:
            self.api_status.config(text="Connected", foreground="green")
        else:
            self.api_status.config(text="Not connected - Enter API key above", foreground="red")
    
    def connect_gemini(self):
        """Connect to Gemini API with provided key"""
        api_key = self.api_key_entry.get().strip()
        if not api_key:
            messagebox.showerror("Error", "Please enter your Gemini API key")
            return
        
        self.log_message("Connecting to Gemini API...")
        self.initialize_gemini(api_key)
        
        if self.use_llm:
            self.api_status.config(text="Connected", foreground="green")
        else:
            self.api_status.config(text="Connection failed", foreground="red")
    
    def speak(self, text: str):
        """Convert text to speech"""
        try:
            self.tts.say(text)
            self.tts.runAndWait()
        except Exception as e:
            self.log_message(f"TTS Error: {e}")
    
    def listen_for_speech(self):
        """Listen for user speech input"""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                self.voice_status.config(text="Listening...")
                self.root.update()
                
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=10)
                
            self.voice_status.config(text="Processing...")
            self.root.update()
            
            text = self.recognizer.recognize_google(audio)
            self.log_message(f"User said: {text}")
            
            return text
            
        except sr.WaitTimeoutError:
            self.log_message("No speech detected within timeout period")
            return None
        except sr.UnknownValueError:
            self.log_message("Could not understand audio")
            return None
        except sr.RequestError as e:
            self.log_message(f"Error with speech recognition: {e}")
            return None
        finally:
            self.voice_status.config(text="Ready to listen...")
            self.root.update()
    
    def extract_locations_with_llm(self, text: str) -> Dict[str, str]:
        """Extract pickup and dropoff locations using Gemini LLM"""
        if not self.use_llm or not self.voice_agent:
            return self.location_extractor.extract_locations(text)
        
        try:
            task = Task(
                description=f"""
                Extract pickup and dropoff locations from this user input: "{text}"
                
                Look for patterns like:
                - "from X to Y"
                - "pick me up at X and drop me at Y"
                - "I want to go from X to Y"
                - "book a cab from X to Y"
                
                Return the result in this exact format:
                PICKUP: [location]
                DROPOFF: [location]
                
                If you cannot find clear pickup and dropoff locations, return:
                ERROR: Could not extract locations
                """,
                agent=self.voice_agent,
                expected_output="Pickup and dropoff locations in the specified format"
            )
            
            crew = Crew(agents=[self.voice_agent], tasks=[task], verbose=False)
            result = crew.kickoff()
            
            # Parse the result
            locations = {'pickup': '', 'dropoff': ''}
            
            if 'ERROR:' in str(result):
                return self.location_extractor.extract_locations(text)
            
            lines = str(result).split('\n')
            for line in lines:
                if 'PICKUP:' in line:
                    locations['pickup'] = line.split('PICKUP:')[1].strip()
                elif 'DROPOFF:' in line:
                    locations['dropoff'] = line.split('DROPOFF:')[1].strip()
            
            # If LLM didn't extract properly, use fallback
            if not locations['pickup'] or not locations['dropoff']:
                return self.location_extractor.extract_locations(text)
            
            return locations
            
        except Exception as e:
            self.log_message(f"LLM extraction failed: {e}")
            return self.location_extractor.extract_locations(text)
    
    def get_prices_direct(self, pickup: str, dropoff: str):
        """Get prices directly from mock API"""
        try:
            uber_price = mock_cab_api.get_price('uber', pickup, dropoff)
            ola_price = mock_cab_api.get_price('ola', pickup, dropoff)
            
            return {
                'uber': uber_price,
                'ola': ola_price
            }
        except Exception as e:
            self.log_message(f"Error getting prices: {e}")
            return None
    
    def get_prices(self, pickup: str, dropoff: str):
        """Get prices using the price retrieval agent or direct API"""
        
        # Always use direct API to avoid LiteLLM issues
        return self.get_prices_direct(pickup, dropoff)
    
    def compare_prices_simple(self, prices: Dict) -> str:
        """Simple price comparison without LLM"""
        uber_price = prices.get('uber', {}).get('price', 0)
        ola_price = prices.get('ola', {}).get('price', 0)
        
        if uber_price < ola_price:
            savings = ola_price - uber_price
            return f"RECOMMENDATION: Uber\nREASON: Cheaper option\nSAVINGS: Rs. {savings:.2f}"
        elif ola_price < uber_price:
            savings = uber_price - ola_price
            return f"RECOMMENDATION: Ola\nREASON: Cheaper option\nSAVINGS: Rs. {savings:.2f}"
        else:
            return f"RECOMMENDATION: Either\nREASON: Same price\nSAVINGS: Rs. 0"
    
    def compare_prices(self, prices: Dict):
        """Compare prices using simple logic (avoiding LiteLLM issues)"""
        return self.compare_prices_simple(prices)
    
    def book_cab_direct(self, platform: str):
        """Book cab directly without agent"""
        if not self.current_pickup or not self.current_dropoff or not self.current_prices:
            return
        
        price = self.current_prices.get(platform, {}).get('price', 0)
        booking = mock_cab_api.book_cab(platform, self.current_pickup, self.current_dropoff, price)
        return json.dumps(booking)
    
    def book_cab(self, platform: str):
        """Book a cab using direct booking (avoiding LiteLLM issues)"""
        if not self.current_pickup or not self.current_dropoff or not self.current_prices:
            self.log_message("No booking information available")
            return
        
        result = self.book_cab_direct(platform)
        self.log_message(f"Booking Result: {result}")
        self.speak(f"Your {platform} cab has been booked successfully")
        
        # Disable booking buttons after successful booking
        self.book_uber_button.config(state='disabled')
        self.book_ola_button.config(state='disabled')
    
    def process_voice_input(self):
        """Process voice input through the agent pipeline"""
        text = self.listen_for_speech()
        if not text:
            return
        
        # Extract locations using LLM or fallback
        locations = self.extract_locations_with_llm(text)
        
        if not locations['pickup'] or not locations['dropoff']:
            self.log_message("Could not extract pickup and dropoff locations. Please try again.")
            self.speak("I couldn't understand the pickup and dropoff locations. Please try again.")
            return
        
        self.current_pickup = locations['pickup']
        self.current_dropoff = locations['dropoff']
        
        self.log_message(f"Pickup: {self.current_pickup}")
        self.log_message(f"Dropoff: {self.current_dropoff}")
        
        # Get prices
        self.log_message("Getting price estimates...")
        prices = self.get_prices(self.current_pickup, self.current_dropoff)
        
        if not prices:
            self.log_message("Could not retrieve prices. Please try again.")
            return
        
        self.current_prices = prices
        
        # Display prices
        uber_price = prices.get('uber', {}).get('price', 0)
        ola_price = prices.get('ola', {}).get('price', 0)
        
        self.log_message(f"Uber price: Rs. {uber_price}")
        self.log_message(f"Ola price: Rs. {ola_price}")
        
        # Compare prices
        comparison = self.compare_prices(prices)
        self.log_message(f"Price Comparison: {comparison}")
        
        # Speak the results
        speech_text = f"Price on Ola is {round(ola_price)} rupees, price on Uber is {round(uber_price)} rupees."
        if ola_price < uber_price:
            speech_text += f" Ola is cheaper by {round(uber_price - ola_price)} rupees. Would you like to book Ola?"
        elif uber_price < ola_price:
            speech_text += f" Uber is cheaper by {round(ola_price - uber_price)} rupees. Would you like to book Uber?"
        else:
            speech_text += " Both platforms have the same price."
        
        self.speak(speech_text)
        
        # Enable booking buttons
        self.book_uber_button.config(state='normal')
        self.book_ola_button.config(state='normal')
    
    def process_manual_input(self):
        """Process manual input"""
        pickup = self.pickup_entry.get().strip()
        dropoff = self.dropoff_entry.get().strip()
        
        if not pickup or not dropoff:
            self.log_message("Please enter both pickup and dropoff locations")
            return
        
        self.current_pickup = pickup
        self.current_dropoff = dropoff
        
        # Get prices
        self.log_message("Getting price estimates...")
        prices = self.get_prices(pickup, dropoff)
        
        if not prices:
            self.log_message("Could not retrieve prices. Please try again.")
            return
        
        self.current_prices = prices
        
        # Display prices
        uber_price = prices.get('uber', {}).get('price', 0)
        ola_price = prices.get('ola', {}).get('price', 0)
        
        self.log_message(f"Uber price: Rs. {uber_price}")
        self.log_message(f"Ola price: Rs. {ola_price}")
        
        # Compare prices
        comparison = self.compare_prices(prices)
        self.log_message(f"Price Comparison: {comparison}")
        
        # Enable booking buttons
        self.book_uber_button.config(state='normal')
        self.book_ola_button.config(state='normal')
    
    def toggle_voice_input(self):
        """Toggle voice input on/off"""
        if self.is_listening:
            self.is_listening = False
            self.voice_button.config(text="Start Voice Input")
            self.voice_status.config(text="Ready to listen...")
        else:
            self.is_listening = True
            self.voice_button.config(text="Stop Voice Input")
            # Start voice input in a separate thread
            threading.Thread(target=self.voice_input_thread, daemon=True).start()
    
    def voice_input_thread(self):
        """Voice input thread to prevent UI blocking"""
        try:
            while self.is_listening:
                self.process_voice_input()
                if self.is_listening:
                    time.sleep(1)  # Small delay between voice inputs
        except Exception as e:
            self.log_message(f"Voice input thread error: {e}")
        finally:
            self.is_listening = False
            self.root.after(0, lambda: self.voice_button.config(text="Start Voice Input"))
            self.root.after(0, lambda: self.voice_status.config(text="Ready to listen..."))

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = VoiceCabApp(root)
    
    # Welcome message
    app.log_message("Welcome to Voice-Enabled Cab Price Comparison!")
    app.log_message("You can use voice input or manual input to compare prices.")
    app.log_message("For voice input, click 'Start Voice Input' and say something like:")
    app.log_message("- 'Book a cab from airport to mall'")
    app.log_message("- 'I want to go from home to office'")
    app.log_message("- 'Pick me up at station and drop me at hospital'")
    app.log_message("")
    app.log_message("For manual input, simply fill in the pickup and dropoff locations.")
    app.log_message("")
    
    # Check if Gemini API key is available
    if not app.use_llm:
        app.log_message("Note: Gemini API not configured. Using simple rule-based extraction.")
        app.log_message("For better location extraction, please add your Gemini API key.")
    
    # Start the GUI event loop
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        # Clean up resources
        try:
            if hasattr(app, 'tts'):
                app.tts.stop()
        except:
            pass
        print("Application closed")

if __name__ == "__main__":
    main()