# -*- coding: utf-8 -*-
def build_query(dataset, **kwargs):
        """Build Query for WRDS database.
        Args:
            dataset (str): dataset
            start (str): Start of time period with format YYYY-MM-DD, default to None
            end (str): End of time period with format YYYY-MM-DD, default to None
            columns (str or list): Filter of one or a list of columns, default to None
            sic (str or list): Filter of one or a list of SIC, default to None
            sich (str or list): Filter of one or a list of historical SIC, default to None
            gvkey (str or list):  Filter of one or a list of Global Company Key, default to None
            tic (str or list):  Filter of one or a list of ticker symbols, default to None
            cusip (str or list):  Filter of one or a list of CUSIP symbols, default to None
            limit (int): Return only a number of return records, defualt to None
        Returns:
            (str) with query
        """
        columns = kwargs.get("columns", None)
        sic = kwargs.get("sic", None)
        sich = kwargs.get("sich", None)
        datadate = kwargs.get("datadate", None)
        start = kwargs.get("start", None)
        end = kwargs.get("end", None)
        gvkey = kwargs.get("gvkey", None)
        tic = kwargs.get("tic", None)
        cusip = kwargs.get("cusip", None)
        permno = kwargs.get("permno", None)
        limit = kwargs.get("limit", None)

        if columns is not None:
            columns_filter = ",".join([x for x in columns])
        else:
            columns_filter = "*"

        date_filter = ""
        if start and end is not None:
            date_filter = datadate +" between {} and {}".format("'" + start + "'" , "'" + end + "'")

        sic_filter = ""
        if sic is not None:
            sic_filter = "sic in ({}) ".format(
                ",".join(["'" + x + "'" for x in sic])
            )

        sich_filter = ""
        if sich is not None:
            sich_filter = "sich in ({}) ".format(
                ",".join(["'" + x + "'" for x in sich])
            )

        gvkey_filter = ""
        if gvkey is not None:
            gvkey_filter = "gvkey in ({}) ".format(
                ",".join(["'" + x + "'" for x in gvkey])
            )

        tic_filter = ""
        if tic is not None:
            tic_filter = "tic in ({}) ".format(
                ",".join(["'" + x + "'" for x in tic])
            )

        cusip_filter = ""
        if cusip is not None:
            cusip_filter = "cusip in ({}) ".format(
                ",".join(["'" + x + "'" for x in cusip])
            )
        
        permno_filter = ""
        if permno is not None:
            permno_filter = "permno in ({}) ".format(
                ",".join(["'" + str(int(x)) + "'" for x in permno])
            )

        # FIRST INSTANCE WITHOUT AND THEN SEQUENTIALLY ADD FILTERS (if non empty)
        filters = [date_filter, cusip_filter, tic_filter, gvkey_filter, sic_filter, permno_filter]
        filters_list = ""
        if any(filters):
            filters_list = " and ".join([x for x in filters if x != ""])
            filters_list = "where " + filters_list

        limit_number = ""
        if limit:
            limit_number += "LIMIT {} ".format(limit)

        cmd = (
            "select "
            + columns_filter
            + " from {} ".format(dataset)
            + filters_list
            + limit_number
            )

        print(cmd)
        return(cmd)