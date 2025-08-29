"""
Mobile App Skeleton for EdgeFedAlign
Note: Requires Kivy to be properly installed for mobile deployment
"""

try:
    from kivy.app import App
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.label import Label
    from kivy.uix.textinput import TextInput
    from kivy.uix.button import Button
    from kivy.uix.scrollview import ScrollView
    KIVY_AVAILABLE = True
except ImportError:
    KIVY_AVAILABLE = False
    print("Kivy not available. Install with: pip install kivy")

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import Agent
from alignment.align_score import AlignmentScorer
from wellbeing.wellbeing_check import WellbeingMonitor


class EdgeFedAlignApp(App if KIVY_AVAILABLE else object):
    def __init__(self, **kwargs):
        if KIVY_AVAILABLE:
            super().__init__(**kwargs)
        self.agent = None
        self.init_agent()
    
    def init_agent(self):
        self.agent = Agent()
        self.agent.set_alignment_module(AlignmentScorer())
        self.agent.set_wellbeing_module(WellbeingMonitor())
    
    def build(self):
        if not KIVY_AVAILABLE:
            return None
            
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        title = Label(
            text='EdgeFedAlign - Privacy-First AI Therapy',
            size_hint=(1, 0.1),
            font_size='20sp'
        )
        main_layout.add_widget(title)
        
        self.output_label = Label(
            text='Welcome! Share your thoughts...',
            size_hint=(1, 0.4),
            text_size=(None, None)
        )
        scroll_view = ScrollView(size_hint=(1, 0.4))
        scroll_view.add_widget(self.output_label)
        main_layout.add_widget(scroll_view)
        
        self.text_input = TextInput(
            multiline=True,
            size_hint=(1, 0.3),
            hint_text='Type your message here...'
        )
        main_layout.add_widget(self.text_input)
        
        button_layout = BoxLayout(size_hint=(1, 0.1), spacing=5)
        
        send_button = Button(text='Send', size_hint=(0.5, 1))
        send_button.bind(on_press=self.on_send)
        button_layout.add_widget(send_button)
        
        clear_button = Button(text='Clear', size_hint=(0.5, 1))
        clear_button.bind(on_press=self.on_clear)
        button_layout.add_widget(clear_button)
        
        main_layout.add_widget(button_layout)
        
        return main_layout
    
    def on_send(self, instance):
        if not KIVY_AVAILABLE:
            return
            
        user_input = self.text_input.text.strip()
        if user_input:
            response = self.agent.run(user_input)
            
            self.output_label.text = f"You: {user_input}\n\nAgent: {response}"
            
            self.text_input.text = ''
    
    def on_clear(self, instance):
        if not KIVY_AVAILABLE:
            return
            
        self.text_input.text = ''
        self.output_label.text = 'Welcome! Share your thoughts...'
    
    def on_start(self):
        if KIVY_AVAILABLE:
            print("EdgeFedAlign mobile app started")
            if self.agent:
                test_response = self.agent.run("mobile test")
                print(f"Agent test response: {test_response}")


def run_mobile_app():
    if not KIVY_AVAILABLE:
        print("=" * 60)
        print("Mobile App - Kivy Not Available")
        print("=" * 60)
        print("\nTo run the mobile app, install Kivy:")
        print("pip install kivy")
        print("\nFor mobile deployment:")
        print("1. Use Buildozer for Android")
        print("2. Use kivy-ios for iOS")
        print("3. Follow platform-specific build instructions")
        return False
    
    app = EdgeFedAlignApp()
    app.run()
    return True


if __name__ == '__main__':
    run_mobile_app()