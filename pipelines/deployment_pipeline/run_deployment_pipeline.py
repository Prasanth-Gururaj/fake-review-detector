import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def run():
    print('--- DEPLOYMENT PIPELINE STARTED ---')
    print('Step 1: Optimize model (Day 4)')
    print('Step 2: Build Docker image (Day 7)')
    print('Step 3: Deploy to EC2 (Day 8)')
    print('--- DEPLOYMENT PIPELINE COMPLETE ---')

if __name__ == '__main__':
    run()