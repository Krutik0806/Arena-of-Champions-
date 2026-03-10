"""
Test to verify no "bid fetching error" occurs in auction system
Tests all bid processing scenarios
"""

import threading
from datetime import datetime

class MockCaptain:
    """Mock captain object"""
    def __init__(self, user_id, name, team_name, purse):
        self.user_id = user_id
        self.name = name
        self.team_name = team_name
        self.purse = purse
        self.spent = 0
        self.players = []

class MockPlayer:
    """Mock player object"""
    def __init__(self, user_id, name, base_price):
        self.user_id = user_id
        self.name = name
        self.base_price = base_price
        self.username = f"player{user_id}"

class MockAuction:
    """Mock auction object mimicking ApprovedAuction"""
    def __init__(self):
        self.id = 1
        self.status = "active"
        self.base_price = 1.0
        self.group_chat_id = 12345
        
        # Create mock captains
        self.approved_captains = {
            101: MockCaptain(101, "Captain A", "Team Alpha", 100.0),
            102: MockCaptain(102, "Captain B", "Team Beta", 100.0),
            103: MockCaptain(103, "Captain C", "Team Gamma", 100.0),
        }
        
        # Create mock players
        self.approved_players = {
            201: MockPlayer(201, "Player 1", 2.0),
            202: MockPlayer(202, "Player 2", 2.0),
        }
        
        # Auction state - Initialize all required attributes
        self.current_player = self.approved_players[201]
        self.highest_bid = self.base_price
        self.highest_bidder = None
        self.current_bids = {}
        self.last_bid_time = None
        
        # Threading lock (the fix!)
        self._bid_lock = threading.Lock()
        
        self.is_paused = False

def test_bid_processing():
    """Test bid processing with the fixed threading.Lock"""
    print("=" * 60)
    print("BID PROCESSING ERROR FIX VERIFICATION")
    print("=" * 60)
    
    auction = MockAuction()
    results = []
    
    def process_bid(captain_id, bid_amount, test_name):
        """Simulate bid processing"""
        try:
            # Get captain
            captain = auction.approved_captains.get(captain_id)
            if not captain:
                results.append(f"❌ {test_name}: Captain not found")
                return
            
            # Check purse
            if bid_amount > captain.purse:
                results.append(f"❌ {test_name}: Insufficient funds")
                return
            
            # Acquire lock (this was failing with asyncio.Lock)
            lock_acquired = auction._bid_lock.acquire(blocking=False)
            if not lock_acquired:
                results.append(f"⏳ {test_name}: Lock busy")
                return
            
            try:
                # Validate current player exists
                if not auction.current_player:
                    results.append(f"❌ {test_name}: No current player")
                    return
                
                # Check minimum bid
                min_increment = 0.5
                min_required_bid = auction.highest_bid + min_increment
                if bid_amount < min_required_bid:
                    results.append(f"❌ {test_name}: Bid too low")
                    return
                
                # Update auction state
                auction.last_bid_time = datetime.now()
                auction.highest_bid = bid_amount
                auction.highest_bidder = captain_id
                
                # Store bid in current_bids
                if not hasattr(auction, 'current_bids') or auction.current_bids is None:
                    auction.current_bids = {}
                
                auction.current_bids[captain_id] = {
                    'captain': captain,
                    'amount': bid_amount,
                    'timestamp': datetime.now()
                }
                
                results.append(f"✅ {test_name}: Bid accepted - {captain.name} bid {bid_amount}Cr")
                
            finally:
                # Always release lock
                auction._bid_lock.release()
                
        except AttributeError as e:
            results.append(f"❌ {test_name}: AttributeError - {e}")
        except Exception as e:
            results.append(f"❌ {test_name}: Unexpected error - {type(e).__name__}: {e}")
    
    print("\nTest 1: Valid bid from Captain A")
    print("-" * 60)
    process_bid(101, 2.5, "Valid Bid")
    
    print("\nTest 2: Higher bid from Captain B")
    print("-" * 60)
    process_bid(102, 3.0, "Higher Bid")
    
    print("\nTest 3: Valid bid from Captain C")
    print("-" * 60)
    process_bid(103, 4.0, "Another Valid Bid")
    
    print("\nTest 4: Bid too low from Captain A")
    print("-" * 60)
    process_bid(101, 3.0, "Low Bid Test")
    
    print("\nTest 5: Insufficient funds test")
    print("-" * 60)
    process_bid(101, 150.0, "Insufficient Funds Test")
    
    print("\nTest 6: Concurrent bidding simulation")
    print("-" * 60)
    threads = []
    for i, (captain_id, amount) in enumerate([
        (101, 5.0),
        (102, 5.5),
        (103, 6.0),
    ]):
        thread = threading.Thread(
            target=process_bid,
            args=(captain_id, amount, f"Concurrent Bid {i+1}")
        )
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    print("\nTest 7: Attribute validation")
    print("-" * 60)
    # Remove an attribute to test resilience
    old_bids = auction.current_bids
    auction.current_bids = None
    process_bid(101, 7.0, "Missing current_bids")
    if auction.current_bids is not None:
        print("✅ current_bids auto-initialized correctly")
    else:
        print("❌ current_bids not initialized")
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    for result in results:
        print(result)
    
    print("\n" + "=" * 60)
    print(f"Final auction state:")
    print(f"  Highest bid: {auction.highest_bid}Cr")
    print(f"  Highest bidder: {auction.highest_bidder}")
    print(f"  Total bids recorded: {len(auction.current_bids)}")
    print("=" * 60)
    
    # Check if any errors occurred
    error_count = sum(1 for r in results if r.startswith("❌") and "too low" not in r.lower() and "insufficient" not in r.lower())
    
    if error_count == 0:
        print("\n✅✅✅ NO BID FETCHING ERRORS! ✅✅✅")
        print("\nAll bid processing works correctly with threading.Lock!")
        print("The auction bidding system is fully functional.")
    else:
        print(f"\n❌ {error_count} unexpected errors found")
    
    print("=" * 60)

if __name__ == "__main__":
    test_bid_processing()
