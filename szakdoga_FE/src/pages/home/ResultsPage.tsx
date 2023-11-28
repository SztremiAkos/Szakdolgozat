import { Box, Button } from "@mui/material";
import { Response } from "../../api/predict";
import { LineChart } from "@mui/x-charts";
import Page from "../../components/Page";

type Props = {
  result: Response;
  selectedSystem: string;
  back: () => void;
};

const ResultsPage = (props: Props) => {
  const { result, selectedSystem, back } = props;
  return (
    <Page title={selectedSystem} r2Score={result.r2Score}>
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "spaceBetween",
        }}
      >
        <Box sx={{ display: "flex" }}>
          <LineChart
            title="Expected"
            series={result.expected.map((ser) => {
              return {
                data: ser,
                showMark: false,
                curve: "natural",
              };
            })}
            width={window.innerWidth * 0.4}
            height={window.innerHeight * 0.6}
          />
          <LineChart
            title="Predicted"
            series={result.predicted.map((ser) => {
              return {
                data: ser,
                showMark: false,
                curve: "natural",
              };
            })}
            width={window.innerWidth * 0.4}
            height={window.innerHeight * 0.6}
          />
        </Box>
        <Box sx={{ paddingLeft: "1.5rem" }}>
          <Button variant="contained" onClick={back}>
            Back
          </Button>
        </Box>
      </Box>
    </Page>
  );
};

export default ResultsPage;
