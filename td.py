from truedata.history import TD_hist
import logging
from dateutil.relativedelta import relativedelta        
from datetime import datetime




td_hist = TD_hist('tdwsp751' ,'raj@751' , log_level= logging.WARNING  )


symbol = 'RELIANCE' 
barsize = '5 min'


def main():
    # # NOTE: Note if market not started / holiday, you will not get data!
	
    # # Gets current day of the specified symbol and bar_size
    # res = td_hist.get_historic_data(symbol, duration="1 D", bar_size='1 min')
    # print(res)
    
    # # Gets last 'n' bars of the specified symbol and bar_size
    # ress = td_hist.get_n_historical_bars(symbol, no_of_bars=30, bar_size= '1 min')
    # print(ress)


    # Get data of the specified timeframe
    res = td_hist.get_historic_data(symbol, start_time=datetime(2025, 9, 28),end_time=datetime(2025, 9, 29), bar_size= 'EOD')
    print(res)
    
    # # Get top gainers/losers of the specified segment
    # res = td_hist.get_gainers("NSEEQ", topn = 25 )
    # print(res)
    # res = td_hist.get_losers("NSEEQ", topn = 25 )
    # print(res)


    # # Get bhavcopy of the specified date and segment. NOTE: If date is not added, it shall return the current day's bhavcopy
    # bhav = td_hist.get_bhavcopy('EQ' , date=datetime(2023, 11, 16)  )
    # print(bhav)




if __name__ == '__main__':
    main()


