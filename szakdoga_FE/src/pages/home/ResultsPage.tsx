import { Box, Button } from "@mui/material";
import { Response } from "../../api/predict";
import { DEFAULT_X_AXIS_KEY, LineChart } from "@mui/x-charts";
import { Label } from "@mui/icons-material";

type Props = {
  result: Response;
  selectedSystem: string;
  back: () => void;
};

const ResultsPage = (props: Props) => {
  const { result, selectedSystem, back } = props;
  return (
    <>
      <p>{selectedSystem}</p>
      <Box sx={{display: 'flex'}}>
        <LineChart 
          series={[
            {
              data: result.expected[0],
              showMark: false,
            },
            {
              data: result.expected[1],
              showMark: false,
            },
          ]}
          width={window.innerWidth *0.4}
          height={window.innerHeight * 0.6}
        />
        <LineChart className="linechart"
          series={[
            {
              data: result.predicted[0],
              showMark: false,
              curve: 'natural'
            },
            {
              data: result.predicted[1],
              showMark: false,
              curve: 'natural'
            },
          ]}
          width={window.innerWidth *0.4}
          height={window.innerHeight * 0.6}
        />
      </Box>
      <Button onClick={back}>Back</Button>
    </>
  );
};

export default ResultsPage;
